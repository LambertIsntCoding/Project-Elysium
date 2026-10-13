"""Make ingested text readable without destroying what it says.

Content reaches Astra's memory from more than one place - a book read through
the background reader (EPUB or plain text), a document, an HTML export, a
model-written sentence - and each source leaves its own debris: HTML tags,
character entities, EPUB ligatures and soft hyphens, zero-width marks, page
furniture, runs of collapsed whitespace and broken line wrapping. Stored raw,
that debris makes a memory unreadable and, worse, unquotable.

This module is the one place that cleans it. It is deliberately *conservative*:

* **It never means to change meaning.** It removes presentation noise and
  repairs whitespace; it does not rewrite, summarise, translate or paraphrase.
* **Code, URLs and identifiers are protected.** Fenced code, inline code spans
  and URLs are lifted out before any transform and restored afterwards, so a
  `&amp;` inside a URL, a `<tag>` inside a snippet, or a full-width character in
  an identifier is left exactly as it was.
* **Paragraphs and sentences survive.** Paragraph breaks are kept (runs of blank
  lines collapse to one blank line, not to nothing); sentence punctuation and
  Unicode text are untouched.
* **It is idempotent.** ``normalize(normalize(x)) == normalize(x)``. A second
  pass is a no-op, so a migration can be re-run without progressively eating the
  content - which is the failure mode the brief warns about.

The module is pure: no I/O, no store, no config. The *application* of it to the
stored corpus lives in :meth:`astra.memory.TripleMemoryStore.normalize_memory_texts`,
which previews before writing and records what it did.
"""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------
# Protected spans: lifted out before any transform, restored after
# ---------------------------------------------------------------------
# A private-use sentinel that no real transform touches and that cannot collide
# with ordinary text. The index is embedded so restoration order is irrelevant.
_OPEN = "\ue000"
_CLOSE = "\ue001"
_PLACEHOLDER_RE = re.compile(_OPEN + r"(\d+)" + _CLOSE)

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"(?:https?://|ftp://|mailto:|www\.)\S+")


def _protect(text: str) -> Tuple[str, List[str]]:
    """Replace code/URL spans with sentinels; return ``(masked, spans)``."""
    spans: List[str] = []

    def _stash(match: "re.Match[str]") -> str:
        spans.append(match.group(0))
        return f"{_OPEN}{len(spans) - 1}{_CLOSE}"

    masked = _FENCED_CODE.sub(_stash, text)
    masked = _INLINE_CODE.sub(_stash, masked)
    masked = _URL.sub(_stash, masked)
    return masked, spans


def _restore(masked: str, spans: List[str]) -> str:
    def _put(match: "re.Match[str]") -> str:
        try:
            return spans[int(match.group(1))]
        except (ValueError, IndexError):
            return match.group(0)

    return _PLACEHOLDER_RE.sub(_put, masked)


# ---------------------------------------------------------------------
# Individual transforms (each idempotent)
# ---------------------------------------------------------------------
# Zero-width and formatting-control characters: invisible presentation noise.
_INVISIBLE_CHARS = (
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\ufeff",  # BOM / zero width no-break space
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u00ad",  # soft hyphen (a break hint, not a real hyphen)
)
_INVISIBLE = dict.fromkeys(map(ord, _INVISIBLE_CHARS))

# The HTML/EPUB tag names worth stripping. Restricting to a known set (rather
# than any ``<...>``) protects genuine angle-bracket prose ("a < b > c") and
# anything the code-protection did not catch.
_TAG_NAMES = (
    "a|abbr|article|aside|b|blockquote|body|br|caption|cite|code|col|colgroup|"
    "dd|del|dfn|div|dl|dt|em|figcaption|figure|footer|h1|h2|h3|h4|h5|h6|head|"
    "header|hr|html|i|img|ins|kbd|li|link|main|meta|nav|ol|p|pre|q|s|samp|"
    "section|small|span|strong|style|sub|sup|table|tbody|td|tfoot|th|thead|"
    "title|tr|u|ul|var"
)
_HTML_TAG = re.compile(
    r"</?(?:" + _TAG_NAMES + r")(?:\s[^<>]{0,300})?/?>", re.IGNORECASE)
_XML_DECL = re.compile(r"<\?[^<>]{0,200}\?>")
_CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)

# A line made only of three or more identical punctuation marks is a decorative
# separator (``***``, ``---``, ``===``, ``___``), not content.
_SEPARATOR_LINE = re.compile(r"^[ \t]*([-*_=~#.]){3,}[ \t]*$", re.MULTILINE)

_ENTITY_HINT = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{1,31}|#\d{1,7}|#x[0-9a-fA-F]{1,6});")
_WHITESPACE_RUN = re.compile(r"[ \t\f\v]+")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_BLANK_RUN = re.compile(r"\n{3,}")

# Sentence-terminal characters: a line ending in one of these is a complete line
# and is never joined to the next (that is where a paragraph or a list ends).
_TERMINAL = set(".!?:;\"')]}>\u201d\u2019\u2026")


def _strip_invisibles(text: str) -> str:
    return text.translate(_INVISIBLE)


def _decode_entities(text: str) -> str:
    """Decode HTML entities to a fixpoint, so the transform is idempotent.

    Only runs when an entity-like token is actually present, so ordinary prose
    (and any stray ``&``) is left alone. Decoding to a fixpoint means a
    double-escaped ``&amp;amp;`` settles at ``&`` in one call and does not keep
    changing on the next pass.
    """
    if not _ENTITY_HINT.search(text):
        return text
    for _ in range(4):  # bounded; entities nest only a few deep in practice
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def _strip_markup(text: str) -> str:
    text = _XML_DECL.sub(" ", text)
    text = _CDATA.sub(r"\1", text)
    return _HTML_TAG.sub(" ", text)


def _nfkc(text: str) -> str:
    """Compatibility-normalise: ligatures (ﬁ -> fi), full-width -> ASCII, etc.

    This is what makes EPUB text (which often carries ``ﬁ``/``ﬂ`` ligatures and
    non-breaking spaces) readable. It is idempotent and does not touch ordinary
    ASCII prose.
    """
    return unicodedata.normalize("NFKC", text)


def _collapse_separators(text: str) -> str:
    return _SEPARATOR_LINE.sub("", text)


def _repair_wrapping(text: str) -> str:
    """Join lines broken mid-sentence by extraction, keeping real breaks.

    A hard-wrapped EPUB line ends without terminal punctuation and the next
    begins with a lower-case letter or digit; that is a broken line, so it is
    rejoined with a space. A line that ends in sentence punctuation, is short
    (a heading, a list item, a fragment), or whose next line begins upper-case
    is left as its own line, so paragraphs, lists and poetry survive.
    """
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        if out and out[-1].strip():
            prev = out[-1].rstrip()
            nxt = line.strip()
            if (len(prev) >= 30
                    and prev[-1] not in _TERMINAL
                    and nxt
                    and (nxt[0].islower() or nxt[0].isdigit())):
                out[-1] = prev + " " + nxt
                continue
        out.append(line)
    return "\n".join(out)


def _collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RUN.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def normalize(text: Any) -> str:
    """Return a cleaned copy of ``text``. Pure and idempotent.

    Protected spans (fenced/inline code and URLs) are preserved verbatim; the
    rest is stripped of markup, entities, invisibles, separators and whitespace
    debris, with ligatures folded to readable characters and broken line wraps
    rejoined. May return ``""`` when the input was entirely markup - callers must
    not persist an empty result.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if not text:
        return ""
    masked, spans = _protect(text)
    masked = _strip_invisibles(masked)
    masked = _decode_entities(masked)
    masked = _strip_markup(masked)
    masked = _nfkc(masked)
    masked = _collapse_separators(masked)
    masked = _collapse_whitespace(masked)
    masked = _repair_wrapping(masked)
    masked = _collapse_whitespace(masked)
    return _restore(masked, spans).strip()


def is_noisy(text: Any) -> bool:
    """Whether :func:`normalize` would change ``text`` at all.

    A cheap gate so a maintenance sweep can skip the (large) majority of
    already-clean records without running the full pipeline on each.
    """
    if not isinstance(text, str) or not text:
        return False
    if _ENTITY_HINT.search(text) or _HTML_TAG.search(text) or _XML_DECL.search(text):
        return True
    if _SEPARATOR_LINE.search(text):
        return True
    if any(ch in text for ch in _INVISIBLE_CHARS):
        return True
    if "\u00a0" in text or "  " in text or "\t" in text:
        return True
    if "\n\n\n" in text:
        return True
    return normalize(text) != text


def diff(text: Any) -> Dict[str, Any]:
    """A before/after view for one string, for preview and audit.

    Reports ``changed`` and whether the cleaned form would be *empty* (which a
    caller must treat as "do not write", not "write nothing").
    """
    before = text if isinstance(text, str) else ("" if text is None else str(text))
    after = normalize(before)
    return {
        "before": before,
        "after": after,
        "changed": after != before,
        "emptied": bool(before.strip()) and not after.strip(),
    }


def audit(texts: Any) -> List[Dict[str, Any]]:
    """Preview the normalization of an iterable of ``(id, text)`` or strings.

    Pure and non-destructive: it reports what *would* change so a migration can
    be shown before it is applied.
    """
    rows: List[Dict[str, Any]] = []
    for item in texts or ():
        if isinstance(item, (tuple, list)) and len(item) == 2:
            ident, value = item
        else:
            ident, value = None, item
        view = diff(value)
        view["id"] = ident
        rows.append(view)
    return rows


__all__ = ["normalize", "is_noisy", "diff", "audit"]
