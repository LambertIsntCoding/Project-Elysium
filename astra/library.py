"""The local library: Astra's books, and where she is in each of them.

Everything here is local and offline. A work is parsed once into a cached plain
text file (``<data_dir>/library/<work_id>.txt``) and a compact reading-state
record is kept beside it (``<data_dir>/reading_state.json``). The reader then
only ever needs the cached text and the offset, so:

* a work can be resumed exactly where it stopped, and an interrupted cycle loses
  at most the chunk it was mid-way through;
* re-reading a work does not require re-parsing the file;
* the expensive part - model inference - is never spent on parsing.

This module owns no model and no memory; it hands the reader text and state, and
records the state back. The *understanding* of a work is stored elsewhere (as
ordinary memories), because that is what has to decay, be contradicted, and be
revised; the *position* is just a cursor.

Storage layout::

    <data_dir>/reading_state.json      # {"version", "queue", "current", "works"}
    <data_dir>/library/<work_id>.txt   # cached extracted text
"""
from __future__ import annotations

import hashlib
import html
import os
import re
import threading
import zipfile
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from . import reading
from .memory import atomic_save, load_json

STATE_VERSION = 1

# Files the library knows how to ingest. ``""`` is included because a
# book-shaped file with no extension is common and is read as plain text.
SUPPORTED_BOOK_EXTENSIONS = (".txt", ".md", ".markdown", ".text", ".epub")
# Default subfolder of the library dir that holds source books. Book filenames
# are used as titles: a space in the name is a space in the title, and " + "
# joins multiple authors (e.g. "Good Omens - Pratchett + Gaiman.md").
DEFAULT_BOOKS_SUBDIR = "books"

# Tags whose text we never want in the extracted plain text.
_SKIP_TAGS = {"script", "style", "head", "title", "meta", "link"}
# Tags that should force a line break so paragraphs survive the strip.
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5",
    "h6", "section", "article", "blockquote", "pre", "hr", "table", "ul", "ol",
}


class _TextExtractor(HTMLParser):
    """Collect visible text from an XHTML document, preserving paragraphs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth and data:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        return _normalise_whitespace(raw)


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of blank lines/spaces while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(str(raw or ""))
        parser.close()
    except Exception:
        # A malformed document must not lose the work: fall back to a crude
        # tag strip rather than failing ingestion.
        stripped = re.sub(r"(?is)<(script|style).*?</\1>", " ", str(raw or ""))
        return _normalise_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", stripped)))
    return parser.text()


def _localname(tag: Any) -> str:
    """The element name without any XML namespace, lower-cased."""
    return str(tag).split("}")[-1].lower()


def _first(elements, name: str):
    for element in elements:
        if _localname(element.tag) == name:
            return element
    return None


def _iter(elements, name: str):
    for element in elements:
        if _localname(element.tag) == name:
            yield element


def epub_to_text(path: str) -> str:
    """Extract readable plain text from an EPUB using only the standard library.

    Follows ``META-INF/container.xml`` to the OPF package, then walks the spine
    in reading order so the text comes out in the order the book intends.
    """
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        container_name = _find_case_insensitive(names, "META-INF/container.xml")
        if container_name is None:
            raise ValueError("not a valid EPUB: META-INF/container.xml is missing")
        container = _parse_xml(archive.read(container_name))
        rootfile = None
        for element in container.iter():
            if _localname(element.tag) == "rootfile":
                rootfile = element.attrib.get("full-path")
                break
        if not rootfile:
            raise ValueError("not a valid EPUB: no rootfile in container.xml")

        opf_name = _find_case_insensitive(names, rootfile) or rootfile
        opf_dir = os.path.dirname(opf_name)
        opf = _parse_xml(archive.read(opf_name))

        manifest: Dict[str, Dict[str, str]] = {}
        for element in opf.iter():
            if _localname(element.tag) == "item":
                item_id = element.attrib.get("id")
                if item_id:
                    manifest[item_id] = {
                        "href": element.attrib.get("href", ""),
                        "media-type": element.attrib.get("media-type", ""),
                    }

        order: List[str] = []
        for element in opf.iter():
            if _localname(element.tag) == "itemref":
                idref = element.attrib.get("idref")
                if idref:
                    order.append(idref)

        if not order:  # no spine: fall back to every content document, sorted
            order = [i for i, item in manifest.items()
                     if "html" in item.get("media-type", "")]

        chunks: List[str] = []
        for item_id in order:
            item = manifest.get(item_id)
            if not item or "html" not in item.get("media-type", ""):
                continue
            href = item["href"].split("#", 1)[0]
            target = _find_case_insensitive(
                names, os.path.normpath(os.path.join(opf_dir, href)) if opf_dir else href
            ) or _find_case_insensitive(names, href)
            if target is None:
                continue
            text = html_to_text(archive.read(target).decode("utf-8", "replace"))
            if text:
                chunks.append(text)
        return _normalise_whitespace("\n\n".join(chunks))


def _find_case_insensitive(names: set, wanted: str) -> Optional[str]:
    wanted = str(wanted or "").replace("\\", "/").lstrip("./")
    if wanted in names:
        return wanted
    lowered = wanted.casefold()
    for name in names:
        if name.casefold() == lowered:
            return name
    return None


def _parse_xml(data: bytes):
    import xml.etree.ElementTree as ET

    return ET.fromstring(data)


def slugify(text: Any) -> str:
    """A stable, filesystem-safe identifier derived from a title or filename."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").casefold()).strip("-")
    return slug[:80] or "work"


def book_title_and_authors(filename: str) -> "tuple[str, List[str]]":
    """Split a book filename into a title and a list of authors.

    The filename is the metadata: the extension is dropped, and a " + " (or a
    lone "+" between words) separates multiple authors. A title containing a
    literal plus is uncommon enough that this is worth the simplicity; a filename
    with no plus is all title and no authors.
    """
    stem = os.path.splitext(os.path.basename(str(filename or "")))[0].strip()
    parts = [p.strip() for p in re.split(r"\s*\+\s*", stem) if p.strip()]
    if len(parts) > 1:
        return parts[0], parts[1:]
    return stem, []


class Library:
    """Astra's local works and her position in each, persisted atomically."""

    def __init__(self, data_dir: str = "./storage", books_path: Optional[str] = None):
        self.data_dir = data_dir
        self.books_dir = os.path.join(data_dir, "library")
        # Where the source books live. Overridable so a scan can point at an
        # arbitrary folder; defaults to ``<data_dir>/library/books``.
        self.books_path = os.path.abspath(
            books_path or os.path.join(self.books_dir, DEFAULT_BOOKS_SUBDIR)
        )
        self.state_file = os.path.join(data_dir, "reading_state.json")
        self._lock = threading.RLock()
        os.makedirs(self.books_dir, exist_ok=True)
        self.state = self._load_state()

    # -- state -------------------------------------------------------------
    @staticmethod
    def _default_state() -> Dict[str, Any]:
        return {"version": STATE_VERSION, "queue": [], "current": "", "works": {}}

    def _load_state(self) -> Dict[str, Any]:
        raw = load_json(self.state_file, dict, self._default_state)
        if not isinstance(raw, dict):
            raw = self._default_state()
        raw.setdefault("version", STATE_VERSION)
        raw.setdefault("queue", [])
        raw.setdefault("current", "")
        works = raw.get("works")
        if not isinstance(works, dict):
            works = {}
        raw["works"] = {str(k): reading.coerce_state(v, str(k)) | _extra_fields(v)
                        for k, v in works.items()}
        queue = [str(q) for q in raw.get("queue", []) if str(q) in raw["works"]]
        raw["queue"] = queue
        if raw.get("current") and raw["current"] not in raw["works"]:
            raw["current"] = ""
        return raw

    def _persist(self) -> None:
        atomic_save(self.state_file, self.state)

    # -- ingestion ---------------------------------------------------------
    def add_text(self, text: str, *, work_id: Optional[str] = None,
                 title: Optional[str] = None, source_format: str = reading.SOURCE_FORMAT_TEXT,
                 source_path: Optional[str] = None, fingerprint: Optional[str] = None,
                 front: bool = False) -> Dict[str, Any]:
        """Register a work from already-extracted text and cache it locally.

        Parsing is done once, here; the reader only ever touches the cached text,
        so an interrupted read never re-parses and never needs the original file.
        """
        text = _normalise_whitespace(str(text or ""))
        if not text:
            raise ValueError("cannot add an empty work")
        with self._lock:
            work_id = self._unique_id(work_id or slugify(title or source_path or "work"))
            cache_path = os.path.join(self.books_dir, work_id + ".txt")
            _write_text_atomic(cache_path, text)
            state = reading.blank_state(
                work_id, title=title or work_id, source_format=source_format,
                total_units=len(text),
                fingerprint=fingerprint or _fingerprint(text),
            )
            state["source_path"] = source_path or os.path.relpath(cache_path, self.data_dir)
            state["started_at"] = _now()
            self.state["works"][work_id] = state
            if front:
                self.state["queue"].insert(0, work_id)
            elif work_id not in self.state["queue"]:
                self.state["queue"].append(work_id)
            if not self.state.get("current"):
                self.state["current"] = work_id
            self._persist()
            return dict(state)

    def add_file(self, path: str, *, work_id: Optional[str] = None,
                 title: Optional[str] = None) -> Dict[str, Any]:
        """Ingest a local file: plain text/markdown, or an EPUB."""
        path = os.path.abspath(str(path or ""))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        ext = os.path.splitext(path)[1].casefold()
        if ext == ".epub":
            text, fmt = epub_to_text(path), reading.SOURCE_FORMAT_EPUB
        elif ext in SUPPORTED_BOOK_EXTENSIONS:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text, fmt = handle.read(), reading.SOURCE_FORMAT_TEXT
        else:
            raise ValueError(f"unsupported file type {ext!r}; use .txt, .md, or .epub")
        inferred_title = title or book_title_and_authors(os.path.basename(path))[0]
        return self.add_text(
            text, work_id=work_id or slugify(inferred_title), title=inferred_title,
            source_format=fmt, source_path=path,
        )

    # -- scanning the books folder ----------------------------------------
    def scan_books(self, folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """List the ingestible books in a folder without ingesting anything.

        Returns one entry per file, in a stable alphabetical order, each with
        ``index``, ``path``, ``filename``, ``title``, ``authors``, ``format``,
        ``size`` and ``ingested`` (already in the library). Read-only: this is
        what makes the numbered picker possible, so it never touches state.
        """
        folder = os.path.abspath(folder or self.books_path)
        if not os.path.isdir(folder):
            return []
        ingested = {str(s.get("source_path") or "") for s in self.state["works"].values()}
        entries: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].casefold()
            if ext not in SUPPORTED_BOOK_EXTENSIONS:
                continue
            title, authors = book_title_and_authors(name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            entries.append({
                "index": len(entries) + 1,
                "path": path,
                "filename": name,
                "title": title,
                "authors": authors,
                "format": ext.lstrip(".") or "text",
                "size": size,
                "ingested": path in ingested,
            })
        return entries

    def add_book(self, reference: Any, folder: Optional[str] = None) -> Dict[str, Any]:
        """Ingest a book by its scan index, its filename, or a full path.

        A bare integer (or a numeric string) selects by the ``index`` from
        :meth:`scan_books`, so a filename containing spaces never has to be
        typed or quoted.
        """
        folder = os.path.abspath(folder or self.books_path)
        text = str(reference or "").strip()
        if not text:
            raise ValueError("no book selected")
        if text.isdigit():
            entries = self.scan_books(folder)
            index = int(text)
            if index < 1 or index > len(entries):
                raise ValueError(f"no book #{index}; {len(entries)} book(s) in {folder}")
            return self.add_file(entries[index - 1]["path"])
        # A filename, or a path. A name with no directory is resolved inside the
        # books folder so the folder never has to be repeated.
        candidate = text if os.path.isabs(text) or os.path.dirname(text) else os.path.join(folder, text)
        return self.add_file(candidate)

    # -- reading -----------------------------------------------------------
    def text_for(self, work_id: str) -> str:
        """The cached plain text of a work (``""`` if it is missing)."""
        with self._lock:
            state = self.state["works"].get(str(work_id))
        if not state:
            return ""
        cache_path = os.path.join(self.books_dir, str(work_id) + ".txt")
        if not os.path.isfile(cache_path):
            return ""
        try:
            with open(cache_path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        except OSError:
            return ""

    def get_state(self, work_id: str) -> Dict[str, Any]:
        with self._lock:
            return reading.coerce_state(self.state["works"].get(str(work_id)))

    def current_work_id(self) -> str:
        with self._lock:
            current = str(self.state.get("current") or "")
            if current and current in self.state["works"]:
                return current
            return self.next_work_id() or ""

    def current_state(self) -> Dict[str, Any]:
        """The reader's current position - the state the reader resumes from."""
        work_id = self.current_work_id()
        if not work_id:
            return reading.blank_state("")
        return self.get_state(work_id)

    def next_work_id(self) -> Optional[str]:
        """The next queued work that still has text left to read."""
        with self._lock:
            for work_id in list(self.state.get("queue", [])):
                state = self.state["works"].get(work_id)
                if state and reading.is_live(state):
                    return work_id
        return None

    def set_current(self, work_id: str) -> None:
        with self._lock:
            if str(work_id) in self.state["works"]:
                self.state["current"] = str(work_id)
                self._persist()

    def save_state(self, work_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Persist an advanced reading-state (the reader's checkpoint)."""
        with self._lock:
            work_id = str(work_id)
            if work_id not in self.state["works"]:
                raise KeyError(work_id)
            merged = reading.coerce_state(state, work_id)
            merged["source_path"] = self.state["works"][work_id].get("source_path", "")
            merged["title"] = merged.get("title") or self.state["works"][work_id].get("title", "")
            self.state["works"][work_id] = merged
            self.state["current"] = work_id
            self._persist()
            return dict(merged)

    def set_status(self, work_id: str, status: str) -> Dict[str, Any]:
        if status not in reading.WORK_STATUSES:
            raise ValueError(f"unknown work status {status!r}")
        with self._lock:
            state = reading.coerce_state(self.state["works"].get(str(work_id)))
            state["status"] = status
            return self.save_state(work_id, state)

    def remove(self, work_id: str) -> bool:
        """Unregister a work and delete its cached text (a deliberate delete)."""
        with self._lock:
            work_id = str(work_id)
            if work_id not in self.state["works"]:
                return False
            self.state["works"].pop(work_id, None)
            self.state["queue"] = [q for q in self.state["queue"] if q != work_id]
            if self.state.get("current") == work_id:
                self.state["current"] = ""
            self._persist()
        cache_path = os.path.join(self.books_dir, work_id + ".txt")
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return True

    # -- views -------------------------------------------------------------
    def list_works(self) -> List[Dict[str, Any]]:
        with self._lock:
            works = [dict(s) for s in self.state["works"].values()]
        works.sort(key=lambda s: (s.get("status") != reading.WORK_READING,
                                  str(s.get("title") or s.get("work_id"))))
        return works

    def _unique_id(self, base: str) -> str:
        base = slugify(base)
        candidate, n = base, 1
        while candidate in self.state["works"]:
            n += 1
            candidate = f"{base}-{n}"
        return candidate


def _extra_fields(raw: Any) -> Dict[str, Any]:
    """Keep the library-only field that ``reading.coerce_state`` does not own."""
    if isinstance(raw, dict) and raw.get("source_path"):
        return {"source_path": str(raw["source_path"])}
    return {"source_path": ""}


def _fingerprint(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _write_text_atomic(path: str, text: str) -> None:
    """Write a text file atomically (temp + replace), like ``atomic_save``."""
    import tempfile

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=os.path.basename(path) + ".", suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "Library", "STATE_VERSION", "slugify", "html_to_text", "epub_to_text",
    "book_title_and_authors", "SUPPORTED_BOOK_EXTENSIONS", "DEFAULT_BOOKS_SUBDIR",
]
