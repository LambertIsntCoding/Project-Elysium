"""The admin menu tree: every slash command reachable without slash syntax.

Built around :class:`astra.admin.MenuRegistry` rather than a branch per command.
Each area registers one screen builder; the registry numbers the top level and
word-aliases each entry, so a command is reachable by number *or* word. Deeper
levels get more specific, exactly mirroring the command surface the CLI already
exposes.

This module is pure wiring: it calls the *existing* session methods
(``_display_memories``, ``_display_self``, ...) and the new live screens. It
adds no new state and never writes a memory.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import admin, inquiry, paths
from . import screens as screens_mod
from .admin import MenuContext, MenuItem, MenuNode


def _run(session: Any, method: str, *args: Any, **kwargs: Any) -> Callable[[MenuContext], Any]:
    """Wrap an existing session method as a menu action."""
    def action(context: MenuContext) -> Any:
        fn = getattr(context.session, method, None)
        if callable(fn):
            fn(*args, **kwargs)
        else:
            context.registry.emit(f"  (this build has no '{method}')")
        return None
    return action


def _paginate_screen(context: MenuContext, records: List[Dict[str, Any]], *,
                     title: str, render: Callable[[Dict[str, Any]], List[str]],
                     permanent: Any = ()) -> None:
    """A reusable newest-first, paged listing with clear navigation.

    One implementation for every listing, so chronological behaviour never
    drifts between commands. Navigation reads like the live screens: ``n`` older,
    ``p`` newer, ``q`` back.
    """
    registry = context.registry
    emit = context.emit
    page = 0
    while True:
        result = admin.paginate(records, page=page)
        emit(f"\n=== {title} ===")
        emit(result.nav_line())
        if result.is_empty:
            emit("  (nothing to show)")
        for mem in result.items:
            for line in render(mem):
                emit(line)
        answer = registry.prompt("  [n] older  [p] newer  [q] back > ")
        token = str(answer or "").strip().casefold()
        if token in ("q", "", "back", "b", "exit"):
            return
        if token in ("n", "next", "older") and result.has_next:
            page += 1
        elif token in ("p", "prev", "newer") and result.has_prev:
            page -= 1
        elif token in ("n", "next", "older"):
            emit("  (already at the oldest page)")
        elif token in ("p", "prev", "newer"):
            emit("  (already at the newest page)")
        else:
            emit("  (unknown key; n=older p=newer q=back)")


# ---------------------------------------------------------------------
# Screen builders
# ---------------------------------------------------------------------
def _screen_conversation(context: MenuContext) -> MenuNode:
    session = context.session
    node = MenuNode("CONVERSATION / SESSION")
    node.items = [
        MenuItem("1", "Session status", aliases=("status", "session"),
                 action=_run(session, "_display_status")),
        MenuItem("2", "Recent conversation", aliases=("history", "recent", "conversation"),
                 action=lambda ctx: _show_history(ctx)),
        MenuItem("3", "Temporary context", aliases=("temporary", "temp"),
                 action=_run(session, "_display_temporary")),
        MenuItem("4", "Sleep log", aliases=("sleep-log", "sleeplog"),
                 action=_run(session, "_display_sleep_log")),
        MenuItem("5", "File decisions", aliases=("decisions", "files"),
                 action=_run(session, "_display_file_decisions")),
        MenuItem("6", "Sleep now", aliases=("sleep",),
                 action=_run(session, "_sleep_now")),
        MenuItem("7", "Wake", aliases=("wake",),
                 action=_run(session, "_wake_now")),
    ]
    return node


def _show_history(context: MenuContext) -> None:
    session = context.session
    history = list(getattr(session, "history", []) or [])
    if not history:
        context.emit("  (no conversation yet this session)")
        return
    for turn in history[-20:]:
        role = str(turn.get("role") or "?")
        content = str(turn.get("content") or "")
        context.emit(f"  [{role}] {content[:300]}")


def _screen_memories(context: MenuContext) -> MenuNode:
    session = context.session
    node = MenuNode("MEMORIES")
    node.items = [
        MenuItem("1", "All memories (overview)", aliases=("all", "overview"),
                 action=lambda ctx: _memory_browser(ctx, "all")),
        MenuItem("2", "Roum memories", aliases=("roum",),
                 action=lambda ctx: _memory_browser(ctx, "roum")),
        MenuItem("3", "Self memories", aliases=("self",),
                 action=lambda ctx: _memory_browser(ctx, "self")),
        MenuItem("4", "Relationship memories", aliases=("relationship",),
                 action=lambda ctx: _memory_browser(ctx, "relationship")),
        MenuItem("5", "Governing", aliases=("governing",),
                 action=_run(session, "_display_governing")),
        MenuItem("6", "Contradictions", aliases=("contradictions",),
                 action=_run(session, "_display_contradictions")),
        MenuItem("7", "Reconcile contradictions", aliases=("reconcile",),
                 action=_run(session, "_reconcile_contradictions")),
        MenuItem("8", "Dormant", aliases=("dormant",),
                 action=_run(session, "_display_dormant")),
        MenuItem("9", "Stats", aliases=("stats",),
                 action=_run(session, "_display_stats")),
        MenuItem("10", "Search", aliases=("search",),
                 action=lambda ctx: _search(ctx)),
        MenuItem("11", "Forget (archive) a memory", aliases=("forget", "archive"),
                 action=lambda ctx: _by_id(ctx, "_archive_memory", "Forget")),
        MenuItem("12", "Restore a memory", aliases=("restore",),
                 action=lambda ctx: _by_id(ctx, "_restore_memory", "Restore")),
        MenuItem("13", "Correct (supersede) a memory", aliases=("correct",),
                 action=_run(session, "_correct_memory")),
        MenuItem("14", "Full record by id", aliases=("memory", "record", "show"),
                 action=lambda ctx: _by_id(ctx, "_display_memory", "Show")),
    ]
    return node


def _memory_browser(context: MenuContext, target: str) -> None:
    """A newest-first paged memory listing, with permanent/observation split."""
    session = context.session
    store = getattr(session, "orchestrator", None)
    store = getattr(store, "store", None)
    targets = ["roum", "self", "relationship"] if target == "all" else [target]
    records: List[Dict[str, Any]] = []
    for model in targets:
        getter = getattr(store, "get_memories", None)
        if callable(getter):
            for mem in getter(model, status=None):
                mem = dict(mem)
                mem.setdefault("target_model", model)
                records.append(mem)

    def render(mem: Dict[str, Any]) -> List[str]:
        status = mem.get("status") or "active"
        tag = f" [{mem.get('target_model')}]" if target == "all" else ""
        return [
            f"   {mem.get('id')}{tag} ({mem.get('type')}) [{status}]",
            f"      {str(mem.get('content') or '')[:200]}",
        ]

    _paginate_screen(context, records, title=f"{target.upper()} MEMORIES", render=render)


def _search(context: MenuContext) -> None:
    query = context.registry.prompt("  search text > ")
    if not str(query or "").strip():
        context.emit("  (cancelled)")
        return
    fn = getattr(context.session, "_search_memories", None)
    if callable(fn):
        fn(str(query))


def _by_id(context: MenuContext, method: str, label: str) -> None:
    ident = context.registry.prompt(f"  {label} id > ")
    if not str(ident or "").strip():
        context.emit("  (cancelled)")
        return
    fn = getattr(context.session, method, None)
    if callable(fn):
        fn(str(ident).strip())


def _screen_self(context: MenuContext) -> MenuNode:
    session = context.session
    node = MenuNode("SELF")
    node.items = [
        MenuItem("1", "Self-portrait (what she is)", aliases=("self", "portrait", "boundary"),
                 action=_run(session, "_display_self")),
        MenuItem("2", "Permanent / settled self-knowledge",
                 aliases=("permanent", "settled", "facts"),
                 action=lambda ctx: _paged_self(ctx, permanent=True)),
        MenuItem("3", "Recent self-observations",
                 aliases=("observations", "recent", "observed"),
                 action=lambda ctx: _paged_self(ctx, permanent=False)),
        MenuItem("4", "Experiences", aliases=("experiences", "experience"),
                 action=_run(session, "_display_experiences")),
    ]
    return node


def _paged_self(context: MenuContext, *, permanent: bool) -> None:
    """Self material, with permanent knowledge and observations kept apart.

    Blending a durable ``self_fact`` with a passing ``self_observation`` is the
    exact confusion the brief calls out, so the two are never on the same page.
    """
    session = context.session
    store = getattr(getattr(session, "orchestrator", None), "store", None)
    getter = getattr(store, "get_memories", None)
    records = getter("self", status=None) if callable(getter) else []
    permanent_types = ("self_fact", "self_preference", "self_belief")
    settled, observed = admin.split_permanent_and_observations(
        records, permanent=permanent_types)
    chosen = settled if permanent else observed
    title = ("PERMANENT SELF-KNOWLEDGE"
             if permanent else "RECENT SELF-OBSERVATIONS")

    def render(mem: Dict[str, Any]) -> List[str]:
        return [
            f"   {mem.get('id')} ({mem.get('type')}) [{mem.get('status')}]",
            f"      {str(mem.get('content') or '')[:200]}",
        ]

    _paginate_screen(context, chosen, title=title, render=render)


def _screen_relationships(context: MenuContext) -> MenuNode:
    node = MenuNode("RELATIONSHIPS & AFFECT")
    node.items = [
        MenuItem("1", "Relationship (live)", aliases=("relationship", "rel", "affinity"),
                 action=lambda ctx: ctx.registry.open_screen("relationship")),
        MenuItem("2", "Relationship diagnostics", aliases=("diagnostics", "diag"),
                 action=_run(ctx_session(context), "_display_relationship")),
        MenuItem("3", "Affect (live)", aliases=("affect", "mood"),
                 action=lambda ctx: ctx.registry.open_screen("affect")),
        MenuItem("4", "Affect diagnostics", aliases=("affect-diag",),
                 action=_run(ctx_session(context), "_display_affect")),
    ]
    return node


def _screen_reading(context: MenuContext) -> MenuNode:
    session = context.session
    node = MenuNode("READING")
    node.items = [
        MenuItem("1", "Reading status", aliases=("reading", "status", "reader"),
                 action=_run(session, "_display_reading")),
        MenuItem("2", "Read one cycle", aliases=("read", "cycle"),
                 action=lambda ctx: _read_control(ctx, [])),
        MenuItem("3", "Read — force now", aliases=("force",),
                 action=lambda ctx: _read_control(ctx, ["force"])),
        MenuItem("4", "Pause reading", aliases=("pause",),
                 action=lambda ctx: _read_control(ctx, ["pause"])),
        MenuItem("5", "Resume reading", aliases=("resume",),
                 action=lambda ctx: _read_control(ctx, ["resume"])),
    ]
    return node


def _read_control(context: MenuContext, args: List[str]) -> None:
    fn = getattr(context.session, "_read_now", None)
    if callable(fn):
        fn(list(args))


def _screen_library(context: MenuContext) -> MenuNode:
    session = context.session
    node = MenuNode("LIBRARY")
    node.items = [
        MenuItem("1", "Library overview", aliases=("library", "list", "works"),
                 action=_run(session, "_display_library")),
        MenuItem("2", "Scan books folder", aliases=("scan", "library-scan"),
                 action=lambda ctx: _library_scan(ctx)),
        MenuItem("3", "Add a book by number", aliases=("add", "add-book"),
                 action=lambda ctx: _add_book(ctx)),
    ]
    return node


def _library_scan(context: MenuContext) -> None:
    fn = getattr(context.session, "_library_scan", None)
    if callable(fn):
        fn(None)


def _add_book(context: MenuContext) -> None:
    number = context.registry.prompt("  book # from the last scan > ")
    if not str(number or "").strip():
        context.emit("  (cancelled)")
        return
    fn = getattr(context.session, "_read_now", None)
    if callable(fn):
        fn(["add", str(number).strip()])


def _screen_questions(context: MenuContext) -> MenuNode:
    session = context.session
    node = MenuNode("QUESTIONS / INQUIRY")
    node.items = [
        MenuItem("1", "Open questions", aliases=("questions", "open", "inquiry"),
                 action=_run(session, "_display_questions")),
        MenuItem("2", "Work knowledge", aliases=("work", "knowledge"),
                 action=lambda ctx: _work(ctx)),
        MenuItem("3", "Prioritised questions (why each is ranked)",
                 aliases=("priority", "prioritised", "ranked"),
                 action=lambda ctx: _question_priority(ctx)),
    ]
    return node


def _work(context: MenuContext) -> None:
    work_id = context.registry.prompt("  work id (blank = current) > ")
    fn = getattr(context.session, "_display_work", None)
    if callable(fn):
        fn(str(work_id).strip() or None)


def _question_priority(context: MenuContext) -> None:
    """Explain which questions are prioritised and why."""
    session = context.session
    store = getattr(getattr(session, "orchestrator", None), "store", None)
    getter = getattr(store, "get_memories", None)
    memories = getter("self", status=None) if callable(getter) else []
    questions = [m for m in memories if inquiry.is_question(m)]
    ranked = inquiry.prioritise_questions(questions)
    context.emit(f"\n=== QUESTION PRIORITY ({len(ranked)} open) ===")
    if not ranked:
        context.emit("  (no open questions)")
        return
    for row in ranked[: admin.PAGE_SIZE]:
        q = row.get("question") or {}
        context.emit(f"  {inquiry.priority_score(q):.3f}  {inquiry.render_question(q)}")
        reasons = row.get("reasons") or []
        for reason in reasons:
            context.emit(f"        - {reason}")


def _screen_timeline(context: MenuContext) -> MenuNode:
    session = context.session
    node = MenuNode("TIMELINE / HISTORY")
    node.items = [
        MenuItem("1", "Timeline (by month)", aliases=("timeline", "history", "time"),
                 action=lambda ctx: _timeline(ctx, ["month"])),
        MenuItem("2", "Timeline (by day)", aliases=("day",),
                 action=lambda ctx: _timeline(ctx, ["day"])),
        MenuItem("3", "Timeline (by year)", aliases=("year",),
                 action=lambda ctx: _timeline(ctx, ["year"])),
        MenuItem("4", "Journal", aliases=("journal",),
                 action=lambda ctx: _journal(ctx, "_display_journal")),
        MenuItem("5", "Reading journal", aliases=("reading-journal",),
                 action=lambda ctx: _journal(ctx, "_display_reading_journal")),
        MenuItem("6", "Sense of time", aliases=("time", "elapsed"),
                 action=_run(session, "_display_time")),
    ]
    return node


def _timeline(context: MenuContext, args: List[str]) -> None:
    fn = getattr(context.session, "_display_timeline", None)
    if callable(fn):
        fn(list(args))


def _journal(context: MenuContext, method: str) -> None:
    period = context.registry.prompt("  period/work id (blank = default) > ")
    fn = getattr(context.session, method, None)
    if callable(fn):
        fn(str(period).strip() or None)


def _screen_elysium(context: MenuContext) -> MenuNode:
    session = context.session
    node = MenuNode("ELYSIUM (ADMINISTRATIVE ROOT)")
    node.items = [
        MenuItem("1", "Elysium status", aliases=("status", "elysium"),
                 action=lambda ctx: _elysium_text(ctx)),
        MenuItem("2", "Elysium help", aliases=("help",),
                 action=lambda ctx: _elysium_text(ctx, "help")),
    ]
    return node


def _elysium_text(context: MenuContext, *args: str) -> None:
    """Elysium stays reachable here, but only as an admin route.

    Elysium is never moved into the model path: this menu only *contains* the
    command, so Astra can never reach or invoke it through conversation.
    """
    session = context.session
    handler = getattr(session, "elysium_handler", None)
    query = " ".join(args) or "status"
    if callable(getattr(handler, "handle", None)):
        try:
            context.emit(str(handler.handle(query)))
            return
        except Exception:
            pass
    context.emit("  (Elysium handler not attached in this session)")


def _screen_diagnostics(context: MenuContext) -> MenuNode:
    session = context.session
    node = MenuNode("DIAGNOSTICS")
    node.items = [
        MenuItem("1", "Debug a message", aliases=("debug",),
                 action=lambda ctx: _debug(ctx)),
    ]
    return node


def _debug(context: MenuContext) -> None:
    message = context.registry.prompt("  message to debug > ")
    fn = getattr(context.session, "_display_debug", None)
    if callable(fn):
        fn(str(message or ""))


def ctx_session(context: MenuContext) -> Any:
    return context.session


# ---------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------
def build_registry(session: Any, *, emit: Optional[Callable[[str], None]] = None,
                   snapshots: Optional[admin.SnapshotStore] = None
                   ) -> admin.MenuRegistry:
    """Build the full admin menu registry for a session.

    ``prompt`` and ``open_screen`` are attached to the registry so menu actions
    (and the live screens) can ask for input or enter a live page without each
    command re-implementing navigation.
    """
    registry = admin.MenuRegistry(session, emit=emit)
    if snapshots is None:
        # Snapshot history is runtime state, not code: keep it under the runtime
        # tree so a Git pull can never touch (or conflict with) it.
        snapshots = admin.SnapshotStore(data_dir=paths.snapshots_dir())
    registry._snapshots = snapshots

    live = screens_mod.build_live_screens(session, registry.snapshots())

    def _prompt(question: str) -> Optional[str]:
        reader = getattr(session, "input_fn", None)
        if callable(reader):
            try:
                return reader(question)
            except (KeyboardInterrupt, EOFError):
                return ""
        return ""

    def _open_screen(name: str) -> None:
        screen = live.get(name)
        if screen is None:
            return
        poll = getattr(session, "_screen_poll", None)
        if not callable(poll):
            # Fall back to a single render when no interactive poll is wired.
            for line in screen.render():
                registry.emit(line)
            return
        screen.run(poll=poll, emit=registry.emit)

    registry.prompt = _prompt  # type: ignore[attr-defined]
    registry.open_screen = _open_screen  # type: ignore[attr-defined]
    registry.live_screens = live  # type: ignore[attr-defined]

    for builder in (_screen_conversation, _screen_memories, _screen_self,
                    _screen_relationships, _screen_reading, _screen_library,
                    _screen_questions, _screen_timeline, _screen_elysium,
                    _screen_diagnostics):
        registry.add_screen(builder)
    return registry


__all__ = ["build_registry", "ctx_session"]
