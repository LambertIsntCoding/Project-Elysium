"""Astra CLI entry point.

Turn flow, in priority order:

1. **Deterministic commands** -- an exact stored trigger is answered directly,
   bypassing the model entirely.
2. **ELYSIUM root layer** -- an input that *invokes* Elysium (the leading word
   ``elysium``) is routed to the application-level :class:`ElysiumCommandHandler`
   before ``build_prompt()`` is ever reached, so Astra's generation path is
   never entered for a root command.
3. **Command registration** -- a message that reads like a conditional command
   is extracted and stored, but only if it validates against the user's own
   words.
4. **Normal turn** -- Astra answers with the assembled prompt, then the turn is
   consolidated into memory.

The session class is separated from the input loop so the routing logic can be
tested without a terminal or a live model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from astra.elysium import ElysiumCommandHandler, ElysiumCommandRecorder, is_elysium_invocation
from astra.memory import CommandStore, TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator

# Phrases that indicate the user is *asking* to register a command. Kept
# deliberately narrow: a bare word like "command" fires on ordinary sentences
# ("what command did I give you?") and would spend a model call for nothing.
_COMMAND_HINTS = ("next time", "when i say", "whenever i say", "from now on when")

_EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}

# Prefixes used only for the display label in the input loop.
_ELYSIUM_LABEL = "Elysium > "
_ASTRA_LABEL = "Astra > "

HELP_TEXT = """
--- CLI COMMANDS ---
 /memories [target] : List memories with key metadata (target optional: roum, self, relationship)
 /memory <id>       : Display full record and provenance for a memory
 /search <text>     : Search memories by content, keywords, or tags
 /governing         : Show the memories that always apply
 /temporary         : Show transient context for this session (not stored)
 /contradictions    : Show memories that were weakened or superseded
 /reconcile         : Link same-subject contradictions that were never resolved
 /dormant           : Show stale memories that have gone dormant
 /stats             : Memory health summary (counts, types, utility)
 /forget <id>       : Retire a memory without deleting it (archive)
 /restore <id>      : Bring an archived memory back
 /correct           : Interactive workflow to supersede an incorrect memory
 /journal           : Display AI journal reflections
 /debug <message>   : Show memory retrieval diagnostics for a message
 /exit              : Save session and exit
--------------------
"""


def looks_like_command_request(text: str) -> bool:
    """True when ``text`` reads like a request to register a conditional command."""
    lowered = text.casefold()
    return any(hint in lowered for hint in _COMMAND_HINTS)


def _default_consolidator(
    user_input: str, ai_response: str, store: TripleMemoryStore,
    conversation_id: str, turn_num: int,
) -> None:
    """Post-turn memory extraction. Failures never interrupt the conversation."""
    try:
        from astra.consolidator import consolidate_turn

        consolidate_turn(user_input, ai_response, store, conversation_id, turn_num)
    except Exception as exc:  # missing/broken consolidator must not crash the loop
        print(f"[consolidation skipped: {exc}]")


class ChatSession:
    """Routes one line of user input to the right handler."""

    def __init__(
        self,
        orchestrator: CompanionOrchestrator,
        cmd_store: CommandStore,
        *,
        elysium: Any = None,
        elysium_handler: Optional[ElysiumCommandHandler] = None,
        recorder: Optional[ElysiumCommandRecorder] = None,
        consolidator: Optional[Callable[..., None]] = None,
        conversation_id: Optional[str] = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ):
        self.orchestrator = orchestrator
        self.cmd_store = cmd_store
        self.elysium = elysium
        # The Elysium root layer always has a handler so an invocation can never
        # fall through into Astra's generation path.
        self.elysium_handler = elysium_handler or ElysiumCommandHandler(elysium)
        self.recorder = recorder
        self.consolidator = consolidator
        self.conversation_id = conversation_id or (
            f"conv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.history: List[Dict[str, str]] = []
        self.turn_counter = 0

    def _emit(self, text: str) -> None:
        self.output_fn(text)

    # -- routing -----------------------------------------------------------
    def handle(self, user_input: str) -> str:
        """Process a non-slash input and return the text to display."""
        # 1. Deterministic command: answer from storage, no model call.
        command_response = self.cmd_store.check_trigger(user_input)
        if command_response:
            self._record_turn(user_input, command_response, consolidate=False)
            return command_response

        # 2. ELYSIUM root-layer routing, decided by the application *before*
        #    Astra's generation path. An invocation never reaches build_prompt().
        if is_elysium_invocation(user_input):
            return self._handle_elysium(user_input)

        # 3. Conditional-command registration (validated before storing).
        if self._try_register_command(user_input):
            return "[System: Command registered.]"

        # 4. Normal conversational turn.
        self.turn_counter += 1
        prompt = self.orchestrator.build_prompt(user_input, self.history)
        response = self.orchestrator.query_gemma(prompt)
        self._record_turn(user_input, response, consolidate=not response.startswith("[Error"))
        # Maintenance runs after the turn's memory writes, never during prompt
        # building (which must stay read-only). It is throttled internally.
        self._run_maintenance()
        return response

    def _run_maintenance(self) -> None:
        """Apply decay/archival between turns, if the store supports it."""
        maintain = getattr(self.orchestrator.store, "maybe_maintain", None)
        if callable(maintain):
            maintain()

    def _handle_elysium(self, user_input: str) -> str:
        """Answer a root invocation without touching Astra or her memory."""
        # Elysium turns are not recorded in Astra's history or consolidated:
        # the root layer is not part of the conversation, and its directives
        # must not be re-learned as Astra memories.
        return self.elysium_handler.handle(user_input)

    def _try_register_command(self, user_input: str) -> bool:
        if self.recorder is None or not looks_like_command_request(user_input):
            return False
        # The recorder only persists commands whose trigger appears in the
        # user's own text, so injected content cannot arm a standing command.
        extracted = self.recorder.capture(user_input)
        if extracted:
            self._emit(f"\n[System: Command registered. Trigger: '{extracted['trigger']}']")
            return True
        return False

    def _record_turn(self, user_input: str, response: str, *, consolidate: bool) -> None:
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        if consolidate and self.consolidator is not None:
            self.consolidator(
                user_input, response, self.orchestrator.store,
                self.conversation_id, self.turn_counter,
            )

    # -- slash commands ----------------------------------------------------
    def _handle_slash(self, user_input: str) -> None:
        parts = user_input.split()
        command = parts[0].lower()

        if command == "/help":
            self._emit(HELP_TEXT)
        elif command == "/memories":
            target = parts[1].lower() if len(parts) > 1 else None
            self._display_memories(target)
        elif command == "/memory":
            if len(parts) > 1:
                self._display_memory(parts[1])
            else:
                self._emit("Usage: /memory <mem_id>")
        elif command == "/search":
            self._search_memories(" ".join(parts[1:]))
        elif command == "/governing":
            self._display_governing()
        elif command == "/temporary":
            self._display_temporary()
        elif command == "/contradictions":
            self._display_contradictions()
        elif command == "/reconcile":
            self._reconcile_contradictions()
        elif command == "/dormant":
            self._display_dormant()
        elif command == "/stats":
            self._display_stats()
        elif command == "/forget":
            if len(parts) > 1:
                self._archive_memory(parts[1])
            else:
                self._emit("Usage: /forget <mem_id>")
        elif command == "/restore":
            if len(parts) > 1:
                self._restore_memory(parts[1])
            else:
                self._emit("Usage: /restore <mem_id>")
        elif command == "/journal":
            self._display_journal()
        elif command == "/debug":
            self._display_debug(" ".join(parts[1:]))
        elif command == "/correct":
            self._correct_memory()
        else:
            self._emit(f"Unknown command: {command}. Type /help for options.")

    def _display_memories(self, target_filter: Optional[str] = None) -> None:
        targets = (
            [target_filter] if target_filter in ("roum", "self", "relationship")
            else ["roum", "self", "relationship"]
        )
        for target in targets:
            self._emit(f"\n=== {target.upper()} MODEL MEMORIES ===")
            memories = self.orchestrator.store.get_memories(target, status=None)
            if not memories:
                self._emit("  (No memories found)")
                continue
            for mem in memories:
                status = mem.get("status", "unknown")
                if status == "superseded":
                    status = f"SUPERSEDED by {mem.get('superseded_by')}"
                self._emit(f" ID: {mem.get('id')} | [{mem.get('type')}] | Status: {status}")
                self._emit(f"    Content: {mem.get('content')}")
                self._emit(
                    f"    Source: {mem.get('source')} | Conf: {mem.get('confidence')}"
                    f" | Turn: {mem.get('originating_turn', '-')}"
                )
                self._emit("-" * 50)

    def _all_memories(self, status: Optional[str] = None):
        """Iterate ``(target, memory)`` across every model."""
        for target in ("roum", "self", "relationship"):
            for mem in self.orchestrator.store.get_memories(target, status=status):
                yield target, mem

    def _search_memories(self, query: str) -> None:
        """Find memories by content, keyword, or tag (case-insensitive)."""
        if not query.strip():
            self._emit("Usage: /search <text>")
            return
        needle = query.casefold()
        hits = []
        for target, mem in self._all_memories(status=None):
            haystack = " ".join([
                str(mem.get("content", "")),
                " ".join(str(k) for k in mem.get("keywords", []) or []),
                " ".join(str(t) for t in mem.get("tags", []) or []),
            ]).casefold()
            if needle in haystack:
                hits.append((target, mem))

        self._emit(f"\n=== SEARCH: {query!r} ({len(hits)} match{'es' if len(hits) != 1 else ''}) ===")
        if not hits:
            self._emit("  (Nothing matched)")
            return
        for target, mem in hits:
            self._emit(f" [{target}] {mem.get('id')} | {mem.get('status')} | {mem.get('type')}")
            self._emit(f"    {mem.get('content')}")

    def _display_governing(self) -> None:
        """The memories that are always injected, regardless of the query."""
        getter = getattr(self.orchestrator.store, "get_governing_memories", None)
        governing = getter() if callable(getter) else []
        self._emit(f"\n=== GOVERNING MEMORIES ({len(governing)} always apply) ===")
        if not governing:
            self._emit("  (None)")
            return
        for mem in governing:
            self._emit(f" [{mem.get('target_model')}] {mem.get('id')} | {mem.get('type')}")
            self._emit(f"    {mem.get('content')}")

    def _display_temporary(self) -> None:
        """Transient scratch for this session only - never written to disk."""
        getter = getattr(self.orchestrator.store, "get_temporary_context", None)
        items = getter(limit=50) if callable(getter) else []
        self._emit(f"\n=== TEMPORARY CONTEXT ({len(items)} item(s), this session only) ===")
        if not items:
            self._emit("  (Nothing transient right now)")
            return
        for item in items:
            self._emit(f" {item.get('id')} | {item.get('kind')} | {item.get('content')}")
        self._emit("  These are NOT stored and will disappear when the session ends.")

    def _display_contradictions(self) -> None:
        """Memories that were weakened or superseded by a later statement."""
        flagged = [
            (target, mem) for target, mem in self._all_memories(status=None)
            if mem.get("contradiction_count") or mem.get("superseded_by")
            or mem.get("contradiction_reason")
        ]
        self._emit(f"\n=== CONTRADICTIONS ({len(flagged)}) ===")
        if not flagged:
            self._emit("  (No contradictions recorded)")
            return
        for target, mem in flagged:
            self._emit(f" [{target}] {mem.get('id')} | status={mem.get('status')}")
            self._emit(f"    {mem.get('content')}")
            if mem.get("superseded_by"):
                self._emit(f"    superseded by: {mem.get('superseded_by')}")
            if mem.get("contradiction_reason"):
                self._emit(f"    reason: {mem.get('contradiction_reason')}")

    def _reconcile_contradictions(self) -> None:
        """Resolve same-subject contradictions the detector previously missed.

        A store learned before the detector recognised restatements can hold
        memories that should already have superseded one another. This replays
        resolution oldest-first, so the newer statement wins; nothing is deleted
        and every superseded record stays readable via /contradictions.
        """
        self._emit("\n--- RECONCILING CONTRADICTIONS ---")
        resolved = self.orchestrator.store.reconcile_contradictions()
        if resolved:
            self._emit(f"✓ Linked {resolved} previously-unresolved contradiction(s).")
            self._emit("  Run /contradictions to review the superseded records.")
        else:
            self._emit("  (No unresolved contradictions found)")

    def _display_dormant(self) -> None:
        """Stale, low-value memories - readable and retrievable, but not governing."""
        dormant = [
            (target, mem) for target, mem in self._all_memories(status=None)
            if str(mem.get("utility") or "active") == "dormant"
        ]
        self._emit(f"\n=== DORMANT MEMORIES ({len(dormant)}) ===")
        if not dormant:
            self._emit("  (None - everything is still active)")
            return
        for target, mem in dormant:
            self._emit(f" [{target}] {mem.get('id')} | {mem.get('type')}")
            self._emit(f"    {mem.get('content')}")
            if mem.get("dormant_reason"):
                self._emit(f"    reason: {mem.get('dormant_reason')}")

    def _display_stats(self) -> None:
        """Memory health summary."""
        stats = self.orchestrator.store.stats()
        self._emit("\n=== MEMORY STATS ===")
        for target in ("roum", "self", "relationship"):
            info = stats.get(target, {})
            self._emit(
                f" {target:>12}: {info.get('active', 0)} active / {info.get('total', 0)} total"
            )
            by_type = info.get("by_type") or {}
            if by_type:
                self._emit(f"      types: {by_type}")
            by_utility = info.get("by_utility") or {}
            if by_utility:
                self._emit(f"      utility: {by_utility}")
            by_status = info.get("by_status") or {}
            if by_status:
                self._emit(f"      status: {by_status}")
        self._emit(f" journal entries: {stats.get('journal_entries', 0)}")

    def _archive_memory(self, mem_id: str) -> None:
        """Retire a memory without deleting it."""
        found = self._find_memory(mem_id)
        if not found:
            self._emit(f"Error: Memory ID '{mem_id}' not found.")
            return
        target, mem = found
        try:
            self.orchestrator.store.archive_memory(target, mem_id, reason="user requested /forget")
        except ValueError as exc:
            self._emit(f"Could not archive: {exc}")
            return
        self._emit(f"\n✓ Memory '{mem_id}' archived (kept on disk, no longer used).")
        self._emit(f"  Was: {mem.get('content')}")

    def _restore_memory(self, mem_id: str) -> None:
        found = self._find_memory(mem_id)
        if not found:
            self._emit(f"Error: Memory ID '{mem_id}' not found.")
            return
        target, _ = found
        try:
            self.orchestrator.store.restore_memory(target, mem_id)
        except ValueError as exc:
            self._emit(f"Could not restore: {exc}")
            return
        self._emit(f"\n✓ Memory '{mem_id}' restored to active.")

    def _find_memory(self, mem_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        for target in ("roum", "self", "relationship"):
            for mem in self.orchestrator.store.get_memories(target, status=None):
                if mem.get("id") == mem_id:
                    return target, mem
        return None

    def _display_memory(self, mem_id: str) -> None:
        found = self._find_memory(mem_id)
        if not found:
            self._emit(f"\nError: Memory ID '{mem_id}' not found in any model.")
            return
        target, mem = found
        self._emit(f"\n=== FULL RECORD: {mem.get('id')} ({target.upper()} MODEL) ===")
        self._emit(json.dumps(mem, indent=2, ensure_ascii=False))

    def _display_journal(self) -> None:
        entries = self.orchestrator.store.get_journal()
        self._emit("\n=== AI MEMORY JOURNAL ===")
        if not entries:
            self._emit("  (No journal reflections recorded yet)")
        for entry in entries:
            self._emit(f"[{entry.get('timestamp')}] {entry.get('title')}")
            self._emit(f"  {entry.get('observation')}\n")

    def _display_debug(self, message: str) -> None:
        """Show which memories were candidates, retrieved, and injected (section 15)."""
        if not message.strip():
            self._emit("Usage: /debug <message>")
            return
        builder = getattr(self.orchestrator, "build_prompt_with_diagnostics", None)
        if not callable(builder):
            self._emit("Diagnostics unavailable: orchestrator does not expose them.")
            return

        _, diag = builder(message, self.history)
        self._emit(f"\n=== MEMORY DIAGNOSTICS for {message!r} ===")
        self._emit(
            f"Boundaries injected: {len(diag['boundaries'])} | "
            f"Governing: {len(diag['governing_ids'])} | "
            f"Current-state slots: {diag['current_state_slots']}"
        )
        self._emit(
            f"Retrieved: {len(diag['retrieved_ids'])} | "
            f"Injected: {len(diag['injected_ids'])} | "
            f"Omitted: {len(diag['omitted_ids'])} | "
            f"Non-retrievable: {len(diag['non_retrievable_ids'])}"
        )
        self._emit("\nCandidates (by score):")
        for cand in diag["candidates"]:
            mark = "INJECTED" if cand["id"] in diag["injected_ids"] else "omitted"
            if not cand["retrievable"]:
                mark = f"non-retrievable ({cand['status']})"
            self._emit(
                f"  [{mark}] {cand['id']} | score={cand['score']} "
                f"| rel={cand['relevance']} | strength={cand['effective_strength']} "
                f"| conf={cand['confidence']} | imp={cand['importance']} "
                f"| src={cand['source']} | status={cand['status']} "
                f"| contradicted={cand['contradicted']}"
            )
            self._emit(f"        {cand['content']}")

    def _correct_memory(self) -> None:
        self._emit("\n--- MEMORY CORRECTION WORKFLOW ---")
        mem_id = self.input_fn("Enter Memory ID to correct: ").strip()
        found = self._find_memory(mem_id)
        if not found:
            self._emit(f"Error: Memory ID '{mem_id}' not found.")
            return

        target, old_mem = found
        self._emit(f'Current Content: "{old_mem.get("content")}"')
        new_content = self.input_fn("Enter CORRECTED statement: ").strip()
        if not new_content:
            self._emit("Correction cancelled (empty input).")
            return
        reason = self.input_fn("Reason for correction (optional): ").strip() or "Explicit user correction"

        new_id = self.orchestrator.store.supersede_memory(
            target_model=target,
            old_id=mem_id,
            content=new_content,
            source="user_correction",
            reason=reason,
        )
        self._emit("\n✓ Memory corrected successfully!")
        self._emit(f"  Old Memory '{mem_id}' -> Marked as 'superseded'")
        self._emit(f"  New Memory '{new_id}' -> Marked as 'active'")

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        self._emit("==========================================================")
        self._emit("  Astra Local Companion Core (Gemma 4 E4B)")
        self._emit("  Type /help for memory inspection & correction commands.")
        self._emit("==========================================================")

        while True:
            try:
                user_input = self.input_fn("\nRoum > ").strip()
            except (KeyboardInterrupt, EOFError):
                self._emit("\nExiting session.")
                break
            if not user_input:
                continue
            if user_input.lower() in _EXIT_WORDS:
                self._emit("Session ended cleanly. Memories saved.")
                break
            if user_input.startswith("/"):
                self._handle_slash(user_input)
                continue
            response = self.handle(user_input)
            label = _ELYSIUM_LABEL if response.startswith(_ELYSIUM_LABEL) else _ASTRA_LABEL
            self._emit(f"\n{label}{response}")


def build_session(
    config_dir: str = "./config",
    storage_dir: str = "./storage",
    *,
    enable_command_extraction: bool = True,
) -> ChatSession:
    """Construct a ready-to-run session with the default components."""
    store = TripleMemoryStore(data_dir=storage_dir)
    cmd_store = CommandStore(data_dir=storage_dir)

    elysium = None
    if os.path.exists(os.path.join(config_dir, "elysium_state.json")):
        from astra.elysium import ElysiumOrchestrator

        elysium = ElysiumOrchestrator(config_dir=config_dir)

    recorder = None
    if enable_command_extraction:
        from astra.elysium import CommandExtractor

        recorder = ElysiumCommandRecorder(cmd_store, CommandExtractor())

    orchestrator = CompanionOrchestrator(
        store, config_dir=config_dir, elysium=elysium, cmd_store=cmd_store,
    )
    return ChatSession(
        orchestrator, cmd_store, elysium=elysium, recorder=recorder,
        consolidator=_default_consolidator,
    )


def main() -> None:
    build_session().run()


if __name__ == "__main__":
    main()
