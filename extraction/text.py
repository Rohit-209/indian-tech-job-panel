"""Text cleaning shared by every extractor.

description_raw is NOT plain text for Greenhouse. The API returns HTML-escaped
HTML (`&lt;p&gt;`), and the collector strips tags before unescaping, so the
stored value still contains real tags. We deliberately don't fix that in the
collector mid-panel: changing description_raw would change every
content_hash on the fix day and fake a wave of edits in the observations.
Instead, extraction cleans defensively here: unescape until stable, then strip.

Boilerplate: every company repeats the same intro, EEO statement and legal
notices across its postings (Okta's NYC Local Law 144 notice mentions
"machine learning" in every single posting). Left in, these inflate a skill's
prevalence in proportion to how much that company hires. `find_boilerplate`
learns them from the corpus: a segment shared by a large share of one
company's postings is boilerplate, not a requirement.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from html import unescape

_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAG = re.compile(r"</?(?:p|div|li|ul|ol|br|h[1-6]|tr|td|section)\b[^>]*>", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9])")
_WS = re.compile(r"\s+")

# A segment is boilerplate if it appears in at least this share of one
# company's postings. Only companies with enough postings to tell template
# from coincidence are checked, and short segments ("Requirements:") are
# ignored because dropping them is harmless either way.
BOILERPLATE_SHARE = 0.40
BOILERPLATE_MIN_POSTINGS = 4
BOILERPLATE_MIN_WORDS = 6


def _unescape_fully(text: str) -> str:
    # Unescape until stable — double-escaped entities (&amp;lt;) do occur.
    for _ in range(3):
        unescaped = unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


def clean_text(raw: str | None) -> str:
    if not raw:
        return ""
    text = _TAG.sub(" ", _unescape_fully(raw)).replace(" ", " ")
    return _WS.sub(" ", text).strip()


def segments(raw: str | None) -> list[str]:
    """Split a description into paragraph/bullet/sentence segments of clean text."""
    if not raw:
        return []
    text = _BLOCK_TAG.sub("\n", _unescape_fully(raw))
    text = _TAG.sub(" ", text).replace(" ", " ")
    out = []
    for block in text.split("\n"):
        for sent in _SENTENCE_END.split(block):
            sent = _WS.sub(" ", sent).strip()
            if sent:
                out.append(sent)
    return out


def segment_key(segment: str) -> str:
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", segment.lower())).strip()


def find_boilerplate(postings: Iterable[tuple[str | None, str | None]]) -> dict[str, set[str]]:
    """Map lowercase company -> set of segment keys that are boilerplate.

    `postings` yields (company, description_raw).
    """
    by_company: dict[str, list[set[str]]] = defaultdict(list)
    for company, desc in postings:
        keys = {
            segment_key(s) for s in segments(desc)
            if len(s.split()) >= BOILERPLATE_MIN_WORDS
        }
        by_company[(company or "").lower()].append(keys)

    result: dict[str, set[str]] = {}
    for company, docs in by_company.items():
        if len(docs) < BOILERPLATE_MIN_POSTINGS:
            continue
        counts = Counter(k for keys in docs for k in keys)
        threshold = BOILERPLATE_SHARE * len(docs)
        result[company] = {k for k, c in counts.items() if c >= threshold}
    return result


def strip_boilerplate(raw: str | None, boilerplate: set[str]) -> str:
    """Clean text of a description with boilerplate segments removed."""
    if not boilerplate:
        return clean_text(raw)
    return " ".join(s for s in segments(raw) if segment_key(s) not in boilerplate)
