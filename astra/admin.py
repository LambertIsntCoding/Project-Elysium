"""The administrative presentation layer: pagination, menus, and live screens.

This module is *presentation only*. It reads the same stores the conversation
path reads, and it renders them for Roum; it never writes a memory, never
touches Astra's prompt, and never routes a message through the model. It exists
because the CLI is now primarily a backend/administrative interface for the
Discord client, and every slash command needs a discoverable, navigable home.

Three reusable pieces live here so individual commands never re-implement them:

* :class:`Paginator` / :func:`sort_newest_first` - one chronological layer. A
  listing is sorted newest-first once, page 0 is the newest page, and later
  pages move *backwards* in time. Empty pages are handled cleanly.
* :class:`MenuNode` / :class:`MenuItem` - a registry-driven menu tree. Every
  entry has a number and word aliases, so the same command is reachable either
  way. The tree is data, not code, so the whole command surface is described in
  one place.
* :class:`LiveScreen` - a continuously refreshing observation page with a tiny
  rolling snapshot history. Snapshots are bounded (``SNAPSHOT_HISTORY``) and are
  deliberately not a log.

The layer is intentionally independent of ``main.py`` so it can be unit-tested
without a running session: it takes plain callables (``emit``, ``read_line``)
rather than reaching for global I/O.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# Tunables. These are the single, easy-to-change knobs for the whole
# administrative layer - page size, snapshot depth, live refresh cadence.
# ---------------------------------------------------------------------

# How many rows one page of any listing shows. Change this one value and every
# paginated command follows.
PAGE_SIZE = 12

# How many previous screen states a live screen keeps. Deliberately tiny: this
# is a rolling diagnostic buffer, never an unbounded log.
SNAPSHOT_HISTORY = 4

# How often a live screen re-renders while it is open.
LIVE_REFRESH_SECONDS = 0.2

# How many snapshots are persisted to disk (a subset of the rolling buffer, so a
# captured state survives the process and can be shown to someone later).
PERSISTED_SNAPSHOTS = SNAPSHOT_HISTORY


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------
# Pagination - one chronological layer for every listing
# ---------------------------------------------------------------------
@dataclass
class Page:
    """One page of a newest-first listing.

    ``page`` is zero-based and counts *backwards in time*: page 0 is the newest
    material, page 1 the next-oldest, and so on. The navigation helpers let a
    caller print an obvious footer without recomputing anything.
    """

    items: List[Dict[str, Any]]
    page: int
    page_size: int
    total: int

    @property
    def total_pages(self) -> int:
        if self.total <= 0:
            return 1
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def has_prev(self) -> bool:
        return self.page > 0

    @property
    def has_next(self) -> bool:
        return self.page + 1 < self.total_pages

    @property
    def is_empty(self) -> bool:
        return not self.items

    @property
    def start_index(self) -> int:
        return self.page * self.page_size

    @property
    def end_index(self) -> int:
        return self.start_index + len(self.items)

    def nav_line(self) -> str:
        """A one-line navigation hint, or an explicit empty-page note."""
        if self.total == 0:
            return "  (nothing to show)"
        oldest_newest = (
            f"showing {self.start_index + 1}-{self.end_index} of {self.total} "
            f"(newest first, page {self.page + 1}/{self.total_pages})"
        )
        parts = [oldest_newest]
        if self.has_prev:
            parts.append("[p] newer")
        if self.has_next:
            parts.append("[n] older")
        parts.append("[q] back")
        return "  " + "  ".join(parts)


def _timestamp_key(mem: Dict[str, Any]) -> str:
    return str(mem.get("timestamp") or mem.get("created_at") or "")


def sort_newest_first(records: Iterable[Dict[str, Any]],
                      key: Optional[Callable[[Dict[str, Any]], Any]] = None
                      ) -> List[Dict[str, Any]]:
    """Sort records newest-first using one shared rule.

    Every chronological listing goes through this, so ordering never drifts
    between commands. The default key is the record timestamp; callers may pass
    a key for records that carry their own time field (e.g. affect history).
    """
    key = key or _timestamp_key
    return sorted(records, key=lambda r: str(key(r) or ""), reverse=True)


def paginate(records: Iterable[Dict[str, Any]], *, page: int = 0,
             page_size: int = PAGE_SIZE, sort: bool = True,
             key: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Page:
    """Slice a listing into a newest-first page.

    ``page`` is clamped into range, so paging past the end lands on the oldest
    page rather than raising or showing a broken view.
    """
    items = sort_newest_first(records, key) if sort else list(records)
    total = len(items)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(int(page), pages - 1))
    start = page * page_size
    return Page(items[start:start + page_size], page, page_size, total)


def split_permanent_and_observations(records: Iterable[Dict[str, Any]],
                                     *, permanent: Sequence[str] = (),
                                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate settled/permanent knowledge from newer observations.

    Self-related material in particular must not blend a durable
    ``self_fact``/``self_preference`` with a passing ``self_observation``; the
    two are shown on their own sections. ``permanent`` names the types that
    count as settled; everything else is treated as an observation.
    """
    permanent_types = set(permanent)
    settled: List[Dict[str, Any]] = []
    observed: List[Dict[str, Any]] = []
    for mem in records:
        if str(mem.get("type") or "") in permanent_types:
            settled.append(mem)
        else:
            observed.append(mem)
    return settled, observed


# ---------------------------------------------------------------------
# Menu registry - the whole command surface described once
# ---------------------------------------------------------------------
@dataclass
class MenuItem:
    """One selectable entry: a number, word aliases, and what it does."""

    key: str
    label: str
    action: Optional[Callable[["MenuContext"], Any]] = None
    aliases: Tuple[str, ...] = ()
    children: Optional["MenuNode"] = None
    help: str = ""

    def matches(self, token: str) -> bool:
        token = str(token or "").strip().casefold()
        if not token:
            return False
        if token == self.key.casefold():
            return True
        return token in {a.casefold() for a in self.aliases}


@dataclass
class MenuNode:
    """A menu screen: a title and its numbered entries."""

    title: str
    items: List[MenuItem] = field(default_factory=list)
    parent: Optional["MenuNode"] = None

    def find(self, token: str) -> Optional[MenuItem]:
        for item in self.items:
            if item.matches(token):
                return item
        return None

    def render(self) -> List[str]:
        lines = [f"\n=== {self.title} ==="]
        for item in self.items:
            alias = f" ({', '.join(item.aliases)})" if item.aliases else ""
            marker = " >" if item.children else ""
            lines.append(f"  {item.key}. {item.label}{alias}{marker}")
        return lines


@dataclass
class MenuContext:
    """Everything a menu action needs, without a global session lookup."""

    session: Any
    emit: Callable[[str], None]
    registry: "MenuRegistry"


class MenuRegistry:
    """Builds and runs the menu tree.

    The registry is the single source of truth for the *structure* of the admin
    UI. Individual screens register a builder that returns a :class:`MenuNode`;
    nothing here hard-codes command behaviour, so adding a screen is one
    registration rather than a new branch in a dispatcher.
    """

    def __init__(self, session: Any, *, emit: Optional[Callable[[str], None]] = None):
        self.session = session
        self._emit = emit or (lambda text: print(text))
        self._root: Optional[MenuNode] = None
        self._builders: List[Callable[[MenuContext], MenuNode]] = []
        self._snapshots = SnapshotStore(
            data_dir=getattr(session, "storage_dir", None) or "./storage")

    def emit(self, text: str) -> None:
        self._emit(text)

    def add_screen(self, builder: Callable[[MenuContext], MenuNode]) -> None:
        self._builders.append(builder)

    def build(self) -> MenuNode:
        context = MenuContext(self.session, self.emit, self)
        root = MenuNode("ASTRA ADMINISTRATION")
        for builder in self._builders:
            node = builder(context)
            if node is not None:
                root.items.append(MenuItem(
                    key=str(len(root.items) + 1), label=node.title,
                    children=node,
                    aliases=(_slug(node.title),),
                ))
        # Exit is always last and always available.
        root.items.append(MenuItem(
            key=str(len(root.items) + 1), label="Exit",
            aliases=("exit", "quit", "q", "x"),
            action=lambda ctx: "exit",
        ))
        self._root = root
        return root

    def root(self) -> MenuNode:
        return self._root or self.build()

    def snapshots(self) -> "SnapshotStore":
        return self._snapshots


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(text).casefold()).strip("-")


# ---------------------------------------------------------------------
# Live screens with a rolling snapshot history
# ---------------------------------------------------------------------
class SnapshotStore:
    """A tiny, bounded rolling snapshot buffer, optionally persisted.

    Deliberately capped: this is a diagnostic aid ("show me the last few states
    of the affect screen"), not a history log. Persisting the newest few lets a
    captured state survive the process so it can be shown to someone later.
    """

    def __init__(self, data_dir: str = "./storage",
                 limit: int = SNAPSHOT_HISTORY) -> None:
        self.path = os.path.join(data_dir, "admin_snapshots.json")
        self.limit = max(1, int(limit))
        self._lock = threading.RLock()
        self._data: Dict[str, List[Dict[str, Any]]] = self._load()

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, List[Dict[str, Any]]] = {}
        for screen, snaps in raw.items():
            if isinstance(snaps, list):
                out[str(screen)] = [s for s in snaps if isinstance(s, dict)][-self.limit:]
        return out

    def _persist(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def capture(self, screen: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Record one snapshot, keeping only the newest ``limit`` per screen."""
        record = dict(snapshot)
        record.setdefault("captured_at", _now())
        with self._lock:
            buffer = self._data.setdefault(str(screen), [])
            buffer.append(record)
            del buffer[:-self.limit]
            self._data[str(screen)] = buffer
            self._persist()
            return record

    def recent(self, screen: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            snaps = list(self._data.get(str(screen), []))
        if limit is not None:
            snaps = snaps[-limit:]
        return snaps


class LiveScreen:
    """A continuously refreshing observation page.

    The screen owns no state of its own beyond its rolling snapshots: it calls
    ``render`` (a pure function of the live stores) on a timer, and ``snapshot``
    to capture the current state into the bounded buffer. It never writes to a
    store and never reaches Astra's prompt.

    Input is injected via ``poll`` (returns a line or ``None`` on timeout), so
    the whole refresh loop is testable without threads or real stdin.
    """

    def __init__(self, name: str, *, render: Callable[[], List[str]],
                 snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
                 store: Optional[SnapshotStore] = None,
                 refresh_seconds: float = LIVE_REFRESH_SECONDS) -> None:
        self.name = name
        self._render = render
        self._snapshot = snapshot
        self.store = store or SnapshotStore()
        self.refresh_seconds = max(0.05, float(refresh_seconds))

    def render(self) -> List[str]:
        return list(self._render())

    def capture(self) -> Dict[str, Any]:
        snapshot = self._snapshot() if callable(self._snapshot) else {"screen": self.name}
        return self.store.capture(self.name, snapshot)

    def history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.store.recent(self.name, limit)

    def run(self, *, poll: Callable[[float], Optional[str]],
            emit: Callable[[str], None], sleep: Callable[[float], None] = time.sleep,
            max_iterations: Optional[int] = None) -> str:
        """Refresh until the user leaves. Returns the exit reason.

        Commands inside a screen: Enter/``r`` refreshes now, ``s`` captures a
        snapshot, ``h`` lists recent snapshots, ``q``/``b``/Esc leaves. Anything
        else is ignored, so a stray key never breaks the loop.
        """
        iterations = 0
        emit(f"\n[Live: {self.name}] refreshing every {self.refresh_seconds:.1f}s. "
             "Enter=refresh  s=snapshot  h=history  q=leave")
        while True:
            for line in self.render():
                emit(line)
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                return "limit"
            command = poll(self.refresh_seconds)
            if command is None:
                continue
            token = str(command).strip().casefold()
            if token in ("", "r", "refresh"):
                continue
            if token in ("q", "b", "quit", "back", "exit", "esc"):
                emit(f"[Left {self.name}]")
                return "left"
            if token in ("s", "snapshot", "capture"):
                snap = self.capture()
                emit(f"  snapshot captured at {snap.get('captured_at')}")
                continue
            if token in ("h", "history", "snapshots"):
                snaps = self.history()
                if not snaps:
                    emit("  (no snapshots yet - press 's' to capture one)")
                for i, snap in enumerate(snaps, 1):
                    emit(f"  {i}. {snap.get('captured_at')}")
                continue
            # Unknown key: ignore rather than crash the live loop.
            emit(f"  (unknown key {command!r}; Enter=refresh  s=snapshot  h=history  q=leave)")


__all__ = [
    "PAGE_SIZE",
    "SNAPSHOT_HISTORY",
    "LIVE_REFRESH_SECONDS",
    "Page",
    "paginate",
    "sort_newest_first",
    "split_permanent_and_observations",
    "MenuItem",
    "MenuNode",
    "MenuContext",
    "MenuRegistry",
    "SnapshotStore",
    "LiveScreen",
]
