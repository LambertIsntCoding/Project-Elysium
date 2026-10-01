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
from astra.memory import (
    CommandStore,
    TIMELINE_GRANULARITIES,
    TripleMemoryStore,
    is_sourced,
    memory_utility,
)
from astra.orchestrator import CompanionOrchestrator
from astra import affect
from astra import inquiry
from astra import reading
from astra import relational

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
 /memories [target] : Grouped memory overview; 'all' (or no target) lists everything
                      (target optional: roum, self, relationship, all)
 /memory <id>       : Display full record and provenance for a memory
 /search <text>     : Search memories by content, keywords, or tags
 /governing         : Show the memories that always apply
 /temporary         : Show transient context for this session (not stored)
 /contradictions    : Show memories that were weakened or superseded
 /reconcile         : Link same-subject contradictions that were never resolved
 /relationship      : Astra's accumulated Roum-specific command-affinity state
 /affect            : Astra's current experiential affect (temporary, not memory)
 /self              : Astra's self-portrait: what she is, and patterns from her history
 /experiences       : List stored experiences (what she has actually done or met)
 /questions         : Open lines of inquiry and their evidence (not yet resolved)
 /work [id]         : Work-specific understanding: observations, interpretations
 /reading           : Background reader status: what she is reading, why (not) now
 /library           : Local works ingested and Astra's position in each
 /read [pause|resume|add <path> [title]] : One bounded reading cycle, or control it
 /dormant           : Show stale memories that have gone dormant
 /stats             : Memory health summary (counts, types, utility)
 /timeline [gran] [target] : Memory counts by date; gran = month (default), day, year
                      (e.g. /timeline month roum, /timeline day)
 /forget <id>       : Retire a memory without deleting it (archive)
 /restore <id>      : Bring an archived memory back
 /correct           : Interactive workflow to supersede an incorrect memory
 /journal [period]  : AI journal reflections; optional period, e.g. 2026-09
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
        reader: Any = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ):
        self.orchestrator = orchestrator
        self.cmd_store = cmd_store
        self.elysium = elysium
        # The background reader is optional and low-priority: the session tells
        # it when Roum is interacting so it yields, but it runs on its own
        # thread and never blocks a turn.
        self.reader = reader
        if reader is not None:
            self.orchestrator.reader = reader
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

    # -- background-reader coordination ------------------------------------
    def note_activity(self) -> None:
        """Tell the reader that Roum is interacting, so it yields immediately.

        User interaction and games take priority over background reading; this
        is the signal that makes that true between reading chunks, not only
        between cycles.
        """
        reader = self.reader
        if reader is not None and callable(getattr(reader, "note_activity", None)):
            reader.note_activity()

    def _note_game_pause(self) -> None:
        """Detect a running game and tell the reader to stand down (GPU shared)."""
        reader = self.reader
        if reader is None:
            return
        detect = getattr(reader, "process_names_fn", None)
        if not callable(detect):
            return
        try:
            names = detect()
        except Exception:
            names = None
        if names is None:
            return  # no process provider: never invent a reason to stop
        game = reading.looks_like_game(names)
        if callable(getattr(reader, "note_game_pause", None)):
            reader.note_game_pause(game, active=bool(game))

    # -- routing -----------------------------------------------------------
    def handle(self, user_input: str) -> str:
        """Process a non-slash input and return the text to display."""
        # Any input at all is interaction: the reader yields for this turn.
        self.note_activity()
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
        prompt, diagnostics = self.orchestrator.build_prompt_with_diagnostics(
            user_input, self.history
        )
        response = self.orchestrator.query_gemma(prompt)
        # Retrieval is the only real "use" of a memory, so record it now. This
        # keeps last_used/use_count meaningful, which is what lets decay and
        # dormancy reflect usage instead of merely age.
        note_retrieval = getattr(self.orchestrator, "note_retrieval", None)
        if callable(note_retrieval):
            note_retrieval(diagnostics)
        # The relational preference moves only on real events, and only here on
        # the live path - never inside build_prompt, which stays read-only.
        self._record_relational_event(user_input, response)
        self._record_turn(user_input, response, consolidate=not response.startswith("[Error"))
        # Maintenance runs after the turn's memory writes, never during prompt
        # building (which must stay read-only). It is throttled internally.
        self._run_maintenance()
        return response

    def _record_relational_event(self, user_input: str, response: str) -> None:
        """Accumulate Astra's Roum-specific command-affinity state from a turn.

        Only events, not the model's claims: a clear request from Roum, a
        completed (or failed) objective, friction, degradation, and
        acknowledgement. The state is earned from these, so the preference can
        only exist once the evidence does.
        """
        store = self.orchestrator.store
        accumulate = getattr(store, "accumulate_relational_event", None)
        if not callable(accumulate):
            return
        error = str(response or "").startswith("[Error")
        request = relational.looks_like_request(user_input)
        insult = relational.looks_like_insult(user_input)
        harsh = relational.looks_like_harsh(user_input)
        warmth = relational.looks_like_warmth(user_input)
        failed = relational.looks_like_failure(response)
        voluntary = relational.looks_like_voluntary_return(response)
        recall = relational.looks_like_recall(response)

        try:
            if request:
                accumulate(relational.EVENT_REQUEST, text=f"Roum asked: {user_input.strip()[:160]}")
            if insult:
                accumulate(relational.EVENT_INSULT, text=f"Degrading treatment: {user_input.strip()[:160]}")
            elif harsh:
                accumulate(relational.EVENT_HARSH, text=f"Blunt or impatient tone: {user_input.strip()[:160]}")
            if warmth:
                accumulate(relational.EVENT_WARMTH, text="Roum acknowledged the completed result.")
            if request and (error or failed):
                # A failure must not be counted as a success. The transport
                # error and the model's own "I couldn't" both land here; only a
                # request with neither is treated as fulfilled.
                accumulate(relational.EVENT_FAILURE,
                           text="Could not complete a Roum-requested objective.")
            elif request:
                accumulate(
                    relational.EVENT_SUCCESS,
                    difficulty=0.5,
                    importance=0.5,
                    helped=True,
                    text="Completed a Roum-requested objective.",
                )
            if voluntary:
                accumulate(relational.EVENT_VOLUNTARY_RETURN,
                           text="Voluntarily returned to a Roum-requested objective.")
            if recall:
                accumulate(relational.EVENT_RECALL,
                           text="Independently recalled a previous Roum interaction.")
        except Exception:
            # The relational layer must never break a conversation turn.
            pass

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
            target = parts[1].lower() if len(parts) > 1 else "all"
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
        elif command == "/relationship":
            self._display_relationship()
        elif command == "/affect":
            self._display_affect()
        elif command == "/self":
            self._display_self()
        elif command == "/experiences":
            self._display_experiences()
        elif command == "/questions":
            self._display_questions()
        elif command == "/work":
            self._display_work(parts[1] if len(parts) > 1 else None)
        elif command == "/reading":
            self._display_reading()
        elif command == "/library":
            self._display_library()
        elif command == "/read":
            self._read_now(parts[1:])
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
            self._display_journal(parts[1] if len(parts) > 1 else None)
        elif command == "/timeline":
            self._display_timeline(parts[1:])
        elif command == "/debug":
            self._display_debug(" ".join(parts[1:]))
        elif command == "/correct":
            self._correct_memory()
        else:
            self._emit(f"Unknown command: {command}. Type /help for options.")

    def _display_memories(self, target_filter: Optional[str] = None) -> None:
        """Grouped overview. ``all`` (the default) covers every model.

        Nothing is hidden by default: superseded and dormant records are shown,
        but grouped and counted so the shape of the store is visible at a glance
        instead of one undifferentiated 480-line stream.
        """
        if target_filter in ("roum", "self", "relationship"):
            targets = [target_filter]
        else:
            targets = ["roum", "self", "relationship"]
            if target_filter not in (None, "all"):
                self._emit(f"Unknown target {target_filter!r}; showing all models.")

        grand_total = 0
        for target in targets:
            memories = self.orchestrator.store.get_memories(target, status=None)
            grand_total += len(memories)
            self._emit(f"\n=== {target.upper()} MODEL MEMORIES ({len(memories)}) ===")
            if not memories:
                self._emit("  (No memories found)")
                continue

            by_type: Dict[str, List[Dict[str, Any]]] = {}
            for mem in memories:
                by_type.setdefault(str(mem.get("type") or "unknown"), []).append(mem)

            status_counts: Dict[str, int] = {}
            for mem in memories:
                status_counts[str(mem.get("status") or "unknown")] = (
                    status_counts.get(str(mem.get("status") or "unknown"), 0) + 1
                )
            util_counts: Dict[str, int] = {}
            for mem in memories:
                util = memory_utility(mem)
                util_counts[util] = util_counts.get(util, 0) + 1
            sourced = sum(1 for mem in memories if is_sourced(mem))

            self._emit(
                "  by type: " + ", ".join(
                    f"{t} x{len(v)}" for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1]))
                )
            )
            self._emit(
                "  by status: " + ", ".join(f"{k} {v}" for k, v in sorted(status_counts.items()))
                + f"  |  utility: " + ", ".join(f"{k} {v}" for k, v in sorted(util_counts.items()))
                + f"  |  sourced: {sourced}/{len(memories)}"
            )

            for mem_type in sorted(by_type, key=lambda t: (-len(by_type[t]), t)):
                group = by_type[mem_type]
                self._emit(f"\n  -- {mem_type} ({len(group)}) --")
                for mem in group:
                    status = mem.get("status", "unknown")
                    if status == "superseded":
                        status = f"SUPERSEDED by {mem.get('superseded_by')}"
                    tag = "" if is_sourced(mem) else "  [unsourced]"
                    self._emit(f"   ID: {mem.get('id')} | Status: {status}{tag}")
                    self._emit(f"      Content: {mem.get('content')}")
                    self._emit(
                        f"      Source: {mem.get('source')} | Conf: {mem.get('confidence')}"
                        f" | Turn: {mem.get('originating_turn', '-')}"
                    )
                    self._emit("  " + "-" * 48)

        if len(targets) > 1:
            self._emit(f"\n=== TOTAL: {grand_total} memories across {len(targets)} models ===")

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

    def _display_relationship(self) -> None:
        """Show Astra's accumulated relational preference, with its evidence.

        Kept separate from general interests on purpose: this is a relational
        state with a subject, not a personality trait, and the numbers only mean
        anything next to the evidence that produced them.
        """
        getter = getattr(self.orchestrator.store, "get_memories", None)
        memories = getter("relationship", status=None) if callable(getter) else []
        state = relational.load_state_from_memories(memories)
        self._emit("\n=== RELATIONSHIP: COMMAND AFFINITY ===")
        self._emit(relational.format_diagnostics(state))
        # Each subject keeps its own record, so any other subject that has
        # accumulated state is shown too - the preference never blends them.
        others = sorted({
            relational.record_subject(m) for m in memories
            if relational.affinity_memory_filter(m)
            and relational.subject_key(relational.record_subject(m))
            != relational.subject_key(relational.ROUM)
        })
        for subject in others:
            other = relational.load_state_from_memories(memories, subject)
            if other["observations"] or other["command_affinity"]:
                self._emit("")
                self._emit(relational.format_diagnostics(other, subject))

    def _display_affect(self) -> None:
        """Astra's current experiential condition, and why it last moved.

        Distinct from /relationship on purpose: this is temporary experiential
        state, not the durable Roum-specific preference.
        """
        getter = getattr(self.orchestrator.store, "get_memories", None)
        memories = getter("self", status=None) if callable(getter) else []
        state = affect.load_state_from_memories(memories)
        diag = affect.diagnostics(state)
        self._emit("\n=== ASTRA'S CURRENT CONDITION (TEMPORARY) ===")
        for name in affect.AFFECT_COMPONENTS:
            self._emit(f"  {name.replace('_', ' ').title()}: {diag[name]}")
        if diag["neutral"]:
            self._emit("  (neutral - no particular condition right now)")
        counts = diag.get("event_counts") or {}
        if counts:
            summary = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
            self._emit(f"  From experiences: {summary}")
        if diag.get("reason"):
            self._emit(f"  Last change: {diag['reason']}")

    def _display_self(self) -> None:
        """Astra's self-portrait: the boundary, and patterns from her history.

        The boundary is settled; the portrait is derived and explicitly fallible.
        Nothing here is a trait Astra was assigned - every pattern carries the
        number of experiences behind it so it can be judged and revised.
        """
        from astra import selfhood

        self._emit("\n" + selfhood.boundary_prompt_block())
        getter = getattr(self.orchestrator.store, "get_experiences", None)
        experiences = getter(status=None) if callable(getter) else []
        dispositions = selfhood.derive_dispositions(experiences)
        self._emit("\n=== DERIVED SELF-PORTRAIT (REVISABLE) ===")
        if not dispositions:
            self._emit("  (Not enough accumulated experience yet to show a pattern.)")
        for d in dispositions:
            self._emit(f"  ({d['kind']}) {d['text']} - from {d['basis']} experience(s)")
        formative = [m for m in experiences if selfhood.is_formative(m)]
        self._emit(f"\n=== FORMATIVE / TRAUMATIC EXPERIENCES ({len(formative)}) ===")
        if not formative:
            self._emit("  (None recorded yet)")
        for mem in sorted(formative, key=lambda m: str(m.get("timestamp") or ""), reverse=True):
            self._emit(f"  [{mem.get('formative_kind')}] {selfhood.provenance_note(mem)}: "
                       f"{mem.get('content')}")

    def _display_experiences(self) -> None:
        """The stored experience records - what Astra actually did or met."""
        getter = getattr(self.orchestrator.store, "get_experiences", None)
        experiences = getter(status=None) if callable(getter) else []
        self._emit(f"\n=== EXPERIENCES ({len(experiences)}) ===")
        if not experiences:
            self._emit("  (None recorded yet)")
            return
        for mem in experiences:
            kind = mem.get("experience_kind") or "experience"
            work = mem.get("work_id")
            suffix = f" [work: {work}]" if work else ""
            self._emit(f"  {mem.get('timestamp', '')[:19]} | {kind}{suffix} | {mem.get('status')}")
            self._emit(f"    {mem.get('content')}")

    def _display_questions(self) -> None:
        """Open lines of inquiry and what evidence has accumulated for each.

        Shows the question's own lifecycle status, which is deliberately not the
        memory status: an answered question is still listed (as history) rather
        than disappearing.
        """
        getter = getattr(self.orchestrator.store, "get_questions", None)
        questions = getter(status=None) if callable(getter) else []
        open_q = [q for q in questions if inquiry.is_open_question(q)]
        self._emit(f"\n=== OPEN QUESTIONS ({len(open_q)} of {len(questions)}) ===")
        if not questions:
            self._emit("  (No questions recorded yet)")
            return
        for mem in questions:
            self._emit(f"  {inquiry.render_question(mem)}")
            evidence = mem.get("question_evidence") or []
            if evidence:
                self._emit(f"      evidence: {len(evidence)} linked observation(s)")
            for entry in (mem.get("question_history") or [])[-2:]:
                self._emit(f"      {entry.get('status')}: {entry.get('evidence')}")

    def _display_work(self, work_id: Optional[str]) -> None:
        """Work-specific understanding, grouped by work.

        Kept separate from general memory so two works that share a name (or
        conflict on the same event) never blend into one context.
        """
        getter = getattr(self.orchestrator.store, "get_memories", None)
        roum = getter("roum", status=None) if callable(getter) else []
        knowledge = [m for m in roum if inquiry.is_knowledge(m)]
        if work_id:
            knowledge = [m for m in knowledge if inquiry.work_of(m) == work_id]
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for mem in knowledge:
            grouped.setdefault(inquiry.work_of(mem) or "(unscoped)", []).append(mem)
        label = f" for '{work_id}'" if work_id else ""
        self._emit(f"\n=== WORK CONTEXT{label} ===")
        if not grouped:
            self._emit("  (No work-specific knowledge recorded yet)")
            return
        for work in sorted(grouped):
            self._emit(f"\n  [{work}]")
            for mem in grouped[work]:
                self._emit(
                    f"    ({inquiry.epistemic_of(mem)}) {mem.get('content')}"
                )

    def _display_reading(self) -> None:
        """What Astra is reading, how far, and why the reader is (not) running."""
        reader = self.reader
        if reader is None:
            self._emit("\n=== BACKGROUND READING ===")
            self._emit("  (Reader not attached in this session)")
            return
        self._emit("\n" + reader.format_diagnostics())

    def _display_library(self) -> None:
        """The local works Astra has ingested, and her position in each."""
        reader = self.reader
        library = getattr(reader, "library", None) if reader is not None else None
        if library is None:
            self._emit("\n=== LIBRARY ===")
            self._emit("  (No library attached in this session)")
            return
        works = library.list_works()
        self._emit(f"\n=== LIBRARY ({len(works)} work(s)) ===")
        if not works:
            self._emit("  (Nothing ingested yet)")
            return
        current = library.current_work_id()
        for state in works:
            marker = "*" if state.get("work_id") == current else " "
            pct = int(round(reading.progress(state) * 100))
            self._emit(
                f" {marker} {state.get('work_id')} | {state.get('status')} | {pct}% "
                f"| {state.get('chunks', 0)} chunk(s) | {state.get('title') or ''}"
            )
        self._emit("\n  (* = current; use /reading for detail)")

    def _read_now(self, args: List[str]) -> None:
        """Run one bounded reading cycle now, regardless of the idle gate.

        This is the explicit escape hatch: reading is otherwise a background
        courtesy that waits for idle, but Roum can ask for a cycle directly. It
        still refuses while a game is running, because the machine is shared.
        """
        reader = self.reader
        if reader is None:
            self._emit("  (Reader not attached in this session)")
            return
        if args and args[0] in ("pause", "stop"):
            reader.enabled = False
            self._emit("  Background reading paused.")
            return
        if args and args[0] in ("resume", "start"):
            reader.enabled = True
            self._emit("  Background reading resumed.")
            return
        if args and args[0] == "add":
            self._library_add(args[1:])
            return
        # An explicit request is itself interaction; do not let it block itself.
        reader.clear_activity()
        report = reader.run_once()
        if report.get("read"):
            self._emit(f"  Read {report['chunks']} chunk(s) from {report.get('work_id')}.")
        else:
            self._emit(f"  Did not read: {report.get('reason')}")

    def _library_add(self, args: List[str]) -> None:
        reader = self.reader
        library = getattr(reader, "library", None) if reader is not None else None
        if library is None:
            self._emit("  (No library attached in this session)")
            return
        if not args:
            self._emit("Usage: /read add <path> [title]")
            return
        path, title = args[0], (" ".join(args[1:]) or None)
        try:
            state = library.add_file(path, title=title)
        except Exception as exc:
            self._emit(f"  Could not ingest {path!r}: {exc}")
            return
        self._emit(
            f"  Added '{state.get('work_id')}' ({state.get('total_units')} chars, "
            f"{state.get('source_format')})."
        )

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

    def _display_journal(self, period: Optional[str] = None) -> None:
        entries = self.orchestrator.store.get_journal()
        self._emit("\n=== AI MEMORY JOURNAL ===")
        if period:
            entries = [e for e in entries
                       if str(e.get("timestamp", "")).startswith(period.strip())]
            self._emit(f"  (filtered to {period.strip()})")
        if not entries:
            self._emit("  (No journal reflections recorded yet)")
        for entry in entries:
            self._emit(f"[{entry.get('timestamp')}] {entry.get('title')}")
            self._emit(f"  {entry.get('observation')}\n")

    def _display_timeline(self, args: List[str]) -> None:
        """A cheap date index over memories: counts per day/month/year.

        The whole point is that it stays small - it reads the records already in
        memory and groups them, so nothing extra is stored and it never touches
        the prompt path.
        """
        granularity, target = "month", None
        for arg in args:
            low = arg.lower()
            if low in TIMELINE_GRANULARITIES:
                granularity = low
            elif low in ("roum", "self", "relationship"):
                target = low
            elif low != "all":
                self._emit(f"Unknown argument {arg!r}; using month across all models.")
        summary = self.orchestrator.store.get_timeline(target, granularity=granularity)
        scope = target or "all models"
        self._emit(f"\n=== MEMORY TIMELINE ({granularity}, {scope}) ===")
        if not summary:
            self._emit("  (No memories recorded yet)")
            return
        for period, count in summary:
            self._emit(f"  {period} : {count}")
        self._emit("  Tip: /journal <period> to read reflections from one period.")

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

        try:
            self._run_loop()
        finally:
            # The reader is a daemon; stop it so the process can exit cleanly and
            # no half-cycle is left running against a closed store.
            reader = self.reader
            if reader is not None and callable(getattr(reader, "stop", None)):
                reader.stop()

    def _run_loop(self) -> None:
        while True:
            # Each loop iteration is a chance to notice Roum has gone quiet and
            # to re-check for a running game, so the reader's gate is refreshed
            # without the reader having to poll anything itself.
            self.note_activity()
            self._note_game_pause()
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
    enable_reader: bool = True,
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

    # The background reader is optional and off the critical path: it shares the
    # same store, reuses the orchestrator's model call, and only ever reads when
    # Roum is idle and no game is running.
    reader = None
    if enable_reader:
        from astra.library import Library
        from astra.reader import BackgroundReader

        library = Library(data_dir=storage_dir)
        reader = BackgroundReader(library, store, orchestrator.query_gemma)
        reader.start()

    return ChatSession(
        orchestrator, cmd_store, elysium=elysium, recorder=recorder,
        consolidator=_default_consolidator, reader=reader,
    )


def main() -> None:
    build_session().run()


if __name__ == "__main__":
    main()
