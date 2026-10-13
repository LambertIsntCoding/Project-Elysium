"""Inspection console for a *running* Astra runtime.

The persistent runtime owns Astra's state and holds the single-instance lock,
so it cannot be opened a second time. This console is the supported way to watch
what is actually happening - and to *drive* it - from a separate process.

How it behaves depends on whether a runtime is running:

* **A runtime is running.** Every command (including mutating ones such as
  ``/sleep``, ``/autonomy pause`` or ``/update apply``) is forwarded over a
  narrow filesystem *control channel* (:mod:`astra.control`) and executed by the
  running runtime through its own ``ApplicationRouter``. There is no second
  implementation here: the console is a genuine remote control, so the whole
  command surface is available and usable, and behaviour cannot drift from the
  CLI or Discord. Conversation is still never routed - only commands are.
* **No runtime is running.** Nothing can be acted on, so the console falls back
  to honest read-only views: the same ``ChatSession`` display methods and
  ``RuntimeController`` reports the live runtime uses, re-read from the durable
  stores before each command. A command that would change state is refused with
  a pointer to the runtime rather than silently doing nothing.

In both modes the console itself holds no authority over Astra's state: it never
builds a second store, never calls the model directly, and never opens Discord.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import control as control_mod
from . import paths

# How long to wait for the running runtime to answer a forwarded command. The
# runtime's control watcher runs on a ~1s cadence, so this is generous; a
# timeout is reported honestly rather than hanging the console.
CONTROL_TIMEOUT_SECONDS = 30.0

# Commands offered by the *runtime* (ApplicationRouter), served read-only by
# calling the controller's report methods and the router's text builders
# directly. Every one of these is a status query (reads a file or an in-memory
# record); none mutates state or contacts the model. The mutating siblings
# (``/sleep``, ``/wake``, ``/autonomy pause``, ``/notifications off``,
# ``/read``, ...) are intentionally absent, and ``router.handle`` is never
# called (its entry point records a human message as a side effect).
ROUTER_INSPECT_COMMANDS: Tuple[str, ...] = (
    "/status", "/tasks", "/schedules", "/scheduler", "/autonomy", "/notifications",
    "/proactive", "/discord", "/power", "/resources", "/proposals", "/decisions",
    "/sleep-log", "/reader", "/normalize-text", "/update",
)

# Session display methods that are pure reads. Each is looked up on the session
# only if it exists, so the console degrades honestly rather than guessing.
SESSION_VIEW_METHODS: Tuple[Tuple[str, str], ...] = (
    ("/memories", "_display_memories"),
    ("/governing", "_display_governing"),
    ("/temporary", "_display_temporary"),
    ("/contradictions", "_display_contradictions"),
    ("/affect", "_display_affect"),
    ("/relationship", "_display_relationship"),
    ("/self", "_display_self"),
    ("/experiences", "_display_experiences"),
    ("/questions", "_display_questions"),
    ("/reading", "_display_reading"),
    ("/library", "_display_library"),
    ("/time", "_display_time"),
    ("/dormant", "_display_dormant"),
    ("/stats", "_display_stats"),
)

# Commands that change something. Recognised only to be refused with a pointer
# to the right place; they are never executed here.
MUTATING_COMMANDS = {
    "/sleep": "the runtime", "/sleep-mode": "the runtime", "/wake": "the runtime",
    "/read": "the runtime", "/read force": "the runtime", "/read pause": "the runtime",
    "/read resume": "the runtime", "/library-scan": "the runtime",
    "/reconcile": "the runtime", "/correct": "the runtime", "/forget": "the runtime",
    "/restore": "the runtime", "/approve": "the runtime", "/reject": "the runtime",
    "/defer": "the runtime", "/autonomy pause": "the runtime",
    "/autonomy resume": "the runtime", "/autonomy now": "the runtime",
    "/notifications pause": "the runtime", "/notifications resume": "the runtime",
    "/notifications mute": "the runtime", "/notifications unmute": "the runtime",
    "/proactive mute": "the runtime", "/proactive unmute": "the runtime",
    "/normalize-text apply": "the runtime",
    "/update apply": "the runtime", "/update reset": "the runtime",
    "/pull apply": "the runtime", "/pull reset": "the runtime",
}

EXIT_WORDS = {"exit", "quit", "q", ":q", "/exit", "/quit"}

# The continuously-refreshing observation pages, rendered read-only here as a
# single frame (no refresh loop, no snapshot writes). Order is the order shown
# by a bare ``/live``.
LIVE_SCREENS: Tuple[str, ...] = ("status", "relationship", "affect")


class _ReadOnlySnapshots:
    """A read-only view over the runtime's captured screen snapshots.

    The live screens read recent snapshots to show direction ("what changed since
    the last capture"). The inspection console may *read* those, but it must
    never write one, so ``capture`` is refused rather than silently persisting.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    def recent(self, screen: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if self._store is None:
            return []
        try:
            return self._store.recent(screen, limit)
        except Exception:
            return []

    def capture(self, screen: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("the inspection console is read-only")


class Inspector:
    """A façade over a live session: a remote control, or a read-only view.

    When a runtime is running, commands are forwarded to it over the control
    channel and executed by the runtime itself. When none is running, the
    console degrades to honest read-only views of the on-disk state.
    """

    def __init__(self, session: Any, *, storage_dir: str, config_dir: str,
                 runtime_dir: Optional[str] = None) -> None:
        self.session = session
        self.controller = getattr(session, "controller", None)
        self.orchestrator = getattr(session, "orchestrator", None)
        self.storage_dir = storage_dir
        self.config_dir = config_dir
        self.runtime_dir = runtime_dir
        self.lock_path = paths.lock_path(runtime_dir)
        self._live: Optional[Dict[str, Any]] = None

    # -- commands --------------------------------------------------------
    def available_commands(self) -> List[str]:
        return sorted({
            *ROUTER_INSPECT_COMMANDS,
            *(name for name, _ in SESSION_VIEW_METHODS),
            "/help", "/report", "/menu", "/live",
        })

    def _known(self, command: str) -> bool:
        if command in ROUTER_INSPECT_COMMANDS:
            return True
        if any(command == name for name, _ in SESSION_VIEW_METHODS):
            return True
        return command in ("/help", "/report", "/live")

    def _mutating(self, command: str, tokens: List[str]) -> Optional[str]:
        if command in MUTATING_COMMANDS:
            return command
        # A rollout command such as "/autonomy pause" is mutating.
        two = " ".join(tokens[:2]).lower()
        if two in MUTATING_COMMANDS:
            return two
        # "/read" with any argument ("/read force") is mutating.
        if command == "/read" and len(tokens) > 1:
            return "/read"
        return None

    # -- read-only snapshots ---------------------------------------------
    def refresh(self) -> None:
        """Re-read the durable stores so the view reflects the live runtime."""
        store = getattr(self.orchestrator, "store", None)
        if store is not None and callable(getattr(store, "reload", None)):
            store.reload()
        controller = self.controller
        if controller is None:
            return
        for owner in (controller,):
            for attr in ("state", "task_store", "ledger", "notifications",
                         "proactive", "proposals", "decisions",
                         "resource_history"):
                obj = getattr(owner, attr, None)
                if obj is not None and callable(getattr(obj, "reload", None)):
                    obj.reload()

    def runtime_status(self) -> Dict[str, Any]:
        """Whether a runtime appears to be running, from the lock file."""
        running = False
        pid: Optional[int] = None
        try:
            with open(self.lock_path, "r", encoding="utf-8") as handle:
                raw = handle.read().strip()
            pid = int(raw) or None
        except (OSError, ValueError):
            pid = None
        if pid:
            try:
                os.kill(pid, 0)
                running = True
            except ProcessLookupError:
                running = False
            except PermissionError:
                running = True
            except OSError:
                running = False
        return {"running": running, "pid": pid, "lock_path": self.lock_path}

    def runtime_listening(self) -> bool:
        """Whether a runtime is running and can receive forwarded commands."""
        return bool(self.runtime_status().get("running"))

    def _forward(self, text: str, *,
                 timeout: float = CONTROL_TIMEOUT_SECONDS) -> Optional[str]:
        """Send a command to the running runtime and return its answer.

        ``None`` means the runtime did not answer in time (or is not listening),
        so the caller can fall back to a read-only view or report honestly.
        """
        request_id = control_mod.submit(text, source="inspect",
                                        runtime_dir=self.runtime_dir)
        result = control_mod.wait_for_result(
            request_id, timeout=timeout, runtime_dir=self.runtime_dir)
        if result is None:
            return None
        return str(result.get("text") or "")

    def snapshot(self) -> str:
        """A truthful, plain-text overview of the live runtime's state."""
        self.refresh()
        rt = self.runtime_status()
        lines = ["=" * 58, "  ASTRA - INSPECTION CONSOLE", "=" * 58]
        if rt["running"]:
            lines.append(
                f"  runtime: RUNNING (pid {rt['pid']})  -  commands are executed "
                "by the running runtime")
        else:
            lines.append(
                "  runtime: not detected  -  read-only view of the state on disk; "
                "start the runtime to act on her")
        controller = self.controller
        if controller is None:
            lines.append("  (persistent runtime not attached)")
            return "\n".join(lines)
        for report in (
            lambda: controller.status_report(),
            controller.autonomy_report,
            lambda: controller.discord_report(getattr(self.session, "discord", None)),
        ):
            try:
                text = report()
            except Exception as exc:  # a view must never take down the console
                text = f"  (unavailable: {exc})"
            if text:
                lines.append("")
                lines.append(text)
        return "\n".join(lines)

    def diagnostics(self) -> Dict[str, Any]:
        """A structured read-only snapshot for tests and programmatic callers."""
        self.refresh()
        controller = self.controller
        out: Dict[str, Any] = {"runtime": self.runtime_status()}
        if controller is None:
            return out
        try:
            out["autonomy"] = controller.autonomy.snapshot()
        except Exception:
            out["autonomy"] = {}
        try:
            out["notifications"] = controller.notifications.snapshot()
        except Exception:
            out["notifications"] = {}
        try:
            out["tasks"] = controller.task_store.counts()
        except Exception:
            out["tasks"] = {}
        discord = getattr(self.session, "discord", None)
        out["discord"] = discord.diagnostics() if discord is not None else None
        return out

    # -- dispatch --------------------------------------------------------
    def help_text(self) -> str:
        if self.runtime_listening():
            lines = ["=" * 58, "  INSPECTION CONSOLE (commands run on the runtime)",
                     "=" * 58]
            lines.append("  A runtime is running, so every command is forwarded to")
            lines.append("  it and executed there - the same behaviour as the CLI or")
            lines.append("  Discord. Conversation is still never routed.")
        else:
            lines = ["=" * 58, "  INSPECTION CONSOLE (read-only; no runtime running)",
                     "=" * 58]
            lines.append("  No runtime is running, so only read-only views are")
            lines.append("  available. Start the runtime (python main.py --supervise)")
            lines.append("  to act on her.")
        lines.append("  /report              Full status snapshot (runtime, autonomy,")
        lines.append("                       Discord, schedules, affect, ...)")
        lines.append("  /menu                The same overview, as a table of sections")
        lines.append("  /live [screen]       Read-only frame of the live screens")
        lines.append("                       (status | relationship | affect; default all)")
        lines.append("  Reports:")
        for command in ROUTER_INSPECT_COMMANDS:
            lines.append(f"    {command}")
        lines.append("  Memory / self / affect views:")
        for command, _ in SESSION_VIEW_METHODS:
            lines.append(f"    {command} [arg]")
        lines.append("  Any other /command is forwarded to the runtime exactly as the")
        lines.append("  console does (mutating commands included, when a runtime is up).")
        lines.append("  exit                 Leave the inspection console")
        return "\n".join(lines)

    def _read_only_commands(self) -> Dict[str, Callable[[], str]]:
        """Direct read-only report callables, keyed by command.

        These are the *same* producers the live runtime uses (the controller's
        report methods and the router's text builders), called directly rather
        than through ``router.handle`` - the router's entry point records a human
        message as a side effect, which inspection must not do.
        """
        controller = self.controller
        router = getattr(self.session, "router", None)
        if controller is None:
            return {}
        commands: Dict[str, Callable[[], str]] = {
            "/status": controller.status_report,
            "/autonomy": controller.autonomy_report,
            "/notifications": controller.autonomy_report,
            "/proactive": lambda: (
                f"Proactive: muted={controller.proactive.muted}. "
                "Mute/unmute from the running runtime."),
            "/power": controller.power_report,
            "/resources": controller.resource_report,
            "/discord": lambda: controller.discord_report(
                getattr(self.session, "discord", None)),
            "/decisions": controller.pending_decisions_text,
            "/sleep-log": controller.sleep_log,
        }
        if router is not None:
            commands["/tasks"] = router._tasks_text
            commands["/schedules"] = router._schedules_text
            commands["/proposals"] = lambda: router._proposals_command([])
        commands["/scheduler"] = lambda: self._scheduler_text()
        commands["/reader"] = controller.reader_health_report
        commands["/normalize-text"] = lambda: controller.normalize_text_report(apply=False)
        commands["/update"] = lambda: controller.update_report(fetch=False)
        return commands

    def _scheduler_text(self) -> str:
        diag = self.controller.scheduler.diagnostics()
        return (f"Scheduler: enabled={diag['enabled']} last={diag['last_reason']} "
                f"counts={diag['task_counts']}")

    # -- live observation screens (read-only) ----------------------------
    def _live_screens(self) -> Dict[str, Any]:
        """The same live screens the admin menu shows, rendered read-only.

        They are the *same* pure renderers the running runtime uses (see
        :mod:`astra.screens`), so what is watched here cannot drift from what the
        menu shows. They read the durable stores and, at most, the runtime's
        captured snapshots for direction - never a model, a thread, or a write.
        """
        if self._live is not None:
            return self._live
        from . import admin, screens

        # The live screens read recent snapshots for direction. Constructing the
        # store only *loads* the file (never writes), and the wrapper refuses a
        # capture, so watching a screen here can never add a snapshot.
        try:
            store = admin.SnapshotStore(data_dir=paths.snapshots_dir())
        except Exception:
            store = None
        snapshots = _ReadOnlySnapshots(store)
        self._live = {
            "status": lambda: screens.status_render(self.session),
            "relationship": lambda: screens.relationship_render(self.session),
            "affect": lambda: screens.affect_render(self.session, snapshots),
        }
        return self._live

    def live_text(self, name: Optional[str] = None) -> str:
        """Render one live screen, or all of them, as a single read-only frame."""
        self.refresh()
        screens = self._live_screens()
        wanted = [name] if name in screens else list(LIVE_SCREENS)
        blocks: List[str] = []
        for screen in wanted:
            render = screens.get(screen)
            if render is None:
                continue
            try:
                blocks.append("\n".join(render()))
            except Exception as exc:
                blocks.append(f"  ({screen} screen unavailable: {exc})")
        header = ("Read-only snapshot - the live screens refresh in the running "
                  "runtime (menu -> Relationships & Affect).")
        return header + "\n\n" + "\n\n".join(blocks) if blocks else header

    def handle(self, line: str) -> str:
        """Resolve one input line: forward to the runtime, or view read-only."""
        text = str(line or "").strip()
        if not text:
            return ""
        tokens = text.split()
        first = tokens[0].lower()
        # Accept a bare word ("status") as an alias for the slash form, but only
        # when it names a real command: a plain sentence must never be mangled
        # into a bogus command, and conversation is never routed at all.
        command = first if first.startswith("/") else "/" + first

        # Console-local meta commands (never forwarded): they are about the
        # console itself, and /report & /live are read-only composites.
        if command in ("/help", "/?"):
            return self.help_text()
        if command == "/report" or command == "/menu":
            return self.snapshot()
        if command == "/live":
            name = tokens[1].lower() if len(tokens) > 1 else None
            return self.live_text(name)

        if not text.startswith("/") and command not in set(self.available_commands()):
            return ("  (the console runs commands only - conversation is never "
                    "sent. Type /help for the options.)")
        # Preserve the original spacing/quoting so a quoted path survives.
        forward_text = text if text.startswith("/") else command + text[len(tokens[0]):]

        # When a runtime is running, the whole command surface is available:
        # forward the command to it and let the runtime execute it through its
        # own router, so there is exactly one implementation of every command.
        if self.runtime_listening():
            answer = self._forward(forward_text)
            if answer is not None:
                return answer
            return ("  (the running runtime did not answer in time; it may be busy "
                    "or restarting - try again in a moment)")

        # No runtime: only honest read-only views, with mutating commands refused.
        refused = self._mutating(command, tokens)
        if refused:
            return (f"  [{refused}] changes Astra's state and is not available "
                    "without a running runtime.\n  Start the runtime "
                    "(python main.py --supervise) to act on her.")

        handler = self._read_only_commands().get(command)
        if handler is not None:
            try:
                return str(handler() or "")
            except Exception as exc:
                return f"  (unavailable: {exc})"

        method_name = dict(SESSION_VIEW_METHODS).get(command)
        if method_name:
            return self._call_view(method_name, tokens[1:])

        return (f"Unknown inspection command: {command}. "
                "Type /help for the options.")

    def _call_view(self, method_name: str, args: List[str]) -> str:
        method = getattr(self.session, method_name, None)
        if not callable(method):
            return f"  (this session does not expose {method_name})"
        buffer: List[str] = []
        original = getattr(self.session, "output_fn", print)
        self.session.output_fn = buffer.append
        try:
            try:
                method(*args)
            except TypeError:
                method()
        except Exception as exc:
            return f"  (view failed: {exc})"
        finally:
            self.session.output_fn = original
        return "\n".join(str(part) for part in buffer if str(part).strip())
