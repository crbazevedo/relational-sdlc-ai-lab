"""Strip explicit cross-references from text so retrieval can't follow the link.

The whole point of the de-referenced benchmark: a relation like ``fixes`` is a
valid *label* but a degenerate *test* while the link is stated in the text
("Fixes #123") — a regex recovers it without any semantics. We keep the link as
ground truth and remove it from the model's input, forcing the model to learn the
semantic issue↔change relationship instead of string-matching an issue number.

``scrub_references`` removes: issue/PR numbers (``#123``, ``gh-123``),
``owner/repo#123``, GitHub issue/PR URLs, and commit SHAs. It deliberately keeps
the surrounding prose (including words like "fixes"/"closes") — those are
semantic content, not the pointer.
"""

from __future__ import annotations

import re

# Order matters: URLs and owner/repo#N before the bare #N rule.
_PATTERNS = [
    # Full GitHub issue / PR / commit URLs.
    re.compile(r"https?://(?:www\.)?github\.com/[\w.\-]+/[\w.\-]+/"
               r"(?:issues|pull|commit)/[\w]+", re.IGNORECASE),
    # owner/repo#123  and  repo#123
    re.compile(r"\b[\w.\-]+/[\w.\-]+#\d+"),
    # #123  and  gh-123
    re.compile(r"(?<![\w])#\d+\b"),
    re.compile(r"\bgh-\d+\b", re.IGNORECASE),
    # Bare commit SHAs (7–40 hex). Require a digit so plain words ("deface") survive.
    re.compile(r"\b(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b", re.IGNORECASE),
]

_LEFTOVER_PARENS = re.compile(r"\(\s*[,;]?\s*\)")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def scrub_references(text: str) -> str:
    """Remove explicit cross-references; keep the surrounding prose."""
    if not text:
        return text or ""
    out = text
    for pat in _PATTERNS:
        out = pat.sub(" ", out)
    out = _LEFTOVER_PARENS.sub(" ", out)
    out = _MULTISPACE.sub(" ", out)
    # Tidy spaces left before punctuation.
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return out.strip()


def scrub_record_text(record: dict) -> str:
    """Return a record's title+body with references scrubbed."""
    c = record.get("content", {})
    title = scrub_references(c.get("title", "") or "")
    body = scrub_references(c.get("body", "") or "")
    return f"{title} {body}".strip()
