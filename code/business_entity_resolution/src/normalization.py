"""
Stage 0 — Normalization.

Mapping-table-driven suffix/abbreviation normalization. Never branches on
`country`, so the same code runs unmodified on US, India, and the unseen
France records in the test set. Landmark phrases ("Near SBI ATM") are
flagged into a separate column rather than deleted. Vectorized with
pandas .str chains (a per-row .map() version was measured at ~16 minutes
on the full multi-million-row corpus; this is far faster).
"""

from __future__ import annotations

import re
import pandas as pd

SUFFIX_MAP = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bllc\b": "llc",
    r"\binc\b": "incorporated",
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\bpllc\b": "pllc",
    r"\bllp\b": "llp",
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\beurl\b": "eurl",
}

ADDRESS_ABBR_MAP = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bbldg\b": "building",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
}

# Outer parentheses around the whole alternation+.* are required so
# str.extract() returns the FULL matched phrase, not just the keyword.
LANDMARK_RE = re.compile(r"((?:near|opp\.?|opposite|behind)\b.*)", flags=re.IGNORECASE)


def _vectorized_clean_punct(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace("+", " plus ", regex=False)
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return s


def normalize_name_series(raw: pd.Series) -> pd.Series:
    text = _vectorized_clean_punct(raw.fillna(""))
    for pattern, repl in SUFFIX_MAP.items():
        text = text.str.replace(pattern, repl, regex=True)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_address_series(raw: pd.Series):
    raw = raw.fillna("").astype(str)
    landmark = raw.str.extract(LANDMARK_RE, expand=False).fillna("").str.strip()
    text = raw.str.replace(LANDMARK_RE, "", regex=True)
    text = _vectorized_clean_punct(text)
    for pattern, repl in ADDRESS_ABBR_MAP.items():
        text = text.str.replace(pattern, repl, regex=True)
    return text.str.replace(r"\s+", " ", regex=True).str.strip(), landmark


def normalize_name(raw: str) -> str:
    """Single-string convenience wrapper, handy for quick tests."""
    return normalize_name_series(pd.Series([raw])).iloc[0]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds business_name_norm, business_address_norm, address_landmark.

    Modifies df IN PLACE (no .copy()) to avoid doubling peak memory across
    several large frames held at once. Pass df.copy() yourself first if
    you need the original untouched.
    """
    df["business_name_norm"] = normalize_name_series(df["business_name"])
    norm_addr, landmark = normalize_address_series(df["business_address"])
    df["business_address_norm"] = norm_addr
    df["address_landmark"] = landmark
    return df


if __name__ == "__main__":
    # Self-test: known tricky pairs. Run this any time normalization changes.
    _test_cases = [
        ("Nyasa Nursing Pvt Ltd", "Nyasa Nursing Private Limited"),
        ("B+ Retail Inc", "B Plus Retail Incorporated"),
        ("Moore Bitwise Inc", "Bitwise Moore LLC"),
        ("Custom Wealth Services LLC", "Custom Wealth Services"),
    ]
    print("Normalization self-test:")
    for a, b in _test_cases:
        na, nb_ = normalize_name(a), normalize_name(b)
        print(f"  {a!r:45s} -> {na!r}")
        print(f"  {b!r:45s} -> {nb_!r}")
        print()
