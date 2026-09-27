"""
Small text-processing helpers shared by Stage 1 (blocking) and Stage 2
(feature engineering), so the two stages can never silently drift apart
(e.g. tokenizing names differently for blocking vs. for the Jaccard
feature).
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_GENERIC_TOKENS = {
    "limited", "private", "llc", "inc", "incorporated", "ltd", "pvt",
    "corp", "corporation", "company", "co", "pllc", "llp", "and",
    "sarl", "sas", "sa", "eurl",
}

_ZIP_RE = re.compile(r"\b\d{5,6}\b")


def tokens(text) -> list:
    """All alnum tokens of length >= 3 (matches blocking's tokenization)."""
    return [t for t in _TOKEN_RE.findall(str(text).lower()) if len(t) >= 3]


def content_tokens(name) -> list:
    """Tokens with generic legal-suffix words (Ltd, Inc, ...) stripped out,
    since those match everywhere and are useless for blocking/Jaccard."""
    toks = [t for t in tokens(name) if t not in _GENERIC_TOKENS]
    return toks if toks else tokens(name)


def token_set(text, min_len: int = 2) -> set:
    """Looser tokenization (min length 2) used for the Jaccard feature,
    where we want a bit more recall on short tokens than blocking does."""
    if not text:
        return set()
    return set(x for x in _TOKEN_RE.findall(str(text).lower()) if len(x) >= min_len)


def extract_zip_pin(text) -> str:
    """Extract the last 5-6 digit numeric run in the text as a likely
    ZIP/PIN code. Taking the *last* match tends to land on the postal
    code rather than a house/street number that appears earlier."""
    if not text:
        return ""
    matches = _ZIP_RE.findall(str(text))
    return matches[-1] if matches else ""


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
