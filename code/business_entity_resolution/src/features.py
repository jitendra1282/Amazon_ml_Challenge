"""
Stage 2 — feature engineering.

Given an S1 row and a candidate S2/S3 row, produce the feature vector the
GBT classifier scores. Used identically during training-pair labeling and
test-set inference, so the two can never drift apart.

Feature list (see config.FEATURE_NAMES for the canonical order):
  0. cosine_sim           - char n-gram TF-IDF cosine (name+address text),
                            using the SAME vectorizer/IDF fit during
                            Stage 1, cached to disk so Stage 2 never
                            re-fits it.
  1. name_jaccard         - token Jaccard overlap on business_name_norm
  2. address_jaccard      - token Jaccard overlap on business_address_norm
  3. zip_exact            - extracted ZIP/PIN codes equal (and both present)
  4. country_exact        - country strings equal (and both present)
  5. name_exact           - normalized names equal (and both present)
  6. name_fuzzy_ratio     - RapidFuzz character ratio on names
  7. address_fuzzy_ratio  - RapidFuzz character ratio on addresses
  8. name_missing         - either normalized name is empty
  9. address_missing      - either normalized address is empty
 10. zip_missing_both     - neither record has an extractable ZIP/PIN
                            (so "zip_exact == 0" here means "unknown", not
                            "known to differ" -- the GBT can learn that
                            distinction instead of us hard-coding it)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio

from text_utils import token_set, extract_zip_pin, jaccard

N_FEATURES = 11


def build_static_features(name1, addr1, country1, name2, addr2, country2) -> list:
    name1 = str(name1) if name1 else ""
    name2 = str(name2) if name2 else ""
    addr1 = str(addr1) if addr1 else ""
    addr2 = str(addr2) if addr2 else ""
    country1 = str(country1).strip().lower() if country1 else ""
    country2 = str(country2).strip().lower() if country2 else ""

    tokens1, tokens2 = token_set(name1), token_set(name2)
    addr_tokens1, addr_tokens2 = token_set(addr1), token_set(addr2)

    zip1, zip2 = extract_zip_pin(addr1), extract_zip_pin(addr2)

    name_missing = float(not name1 or not name2)
    address_missing = float(not addr1 or not addr2)
    zip_missing_both = float(not zip1 and not zip2)

    return [
        jaccard(tokens1, tokens2),
        jaccard(addr_tokens1, addr_tokens2),
        float(bool(zip1) and bool(zip2) and zip1 == zip2),
        float(bool(country1) and bool(country2) and country1 == country2),
        float(bool(name1) and bool(name2) and name1 == name2),
        ratio(name1, name2) / 100.0,
        ratio(addr1, addr2) / 100.0,
        name_missing,
        address_missing,
        zip_missing_both,
    ]


def create_feature_batch(s1_ids, cand_ids, s1_df, s23_df, s1_idx, s23_idx, vectorizer, idf):
    """Featurize a batch of (s1_id, candidate_id) pairs.

    Returns (X, valid_s1_ids, valid_candidate_ids) -- pairs whose IDs
    aren't found in the given frames are silently dropped (this can
    legitimately happen if a candidate file references an ID from a
    different split).
    """
    from blocking import _combined_text  # local import: avoids a cycle at module load

    s1_positions = s1_idx.get_indexer(s1_ids)
    cand_positions = s23_idx.get_indexer(cand_ids)

    valid = [i for i, (sp, cp) in enumerate(zip(s1_positions, cand_positions)) if sp >= 0 and cp >= 0]

    if not valid:
        return np.empty((0, N_FEATURES), dtype=np.float32), [], []

    real_s1_ids = [s1_ids[i] for i in valid]
    real_cand_ids = [cand_ids[i] for i in valid]

    s1_rows = s1_df.iloc[[s1_positions[i] for i in valid]]
    cand_rows = s23_df.iloc[[cand_positions[i] for i in valid]]

    # ---- cosine similarity (reuses the Stage 1 vectorizer/IDF) ----
    s1_text = _combined_text(s1_rows)
    cand_text = _combined_text(cand_rows)
    s1_text.index = range(len(s1_text))
    cand_text.index = range(len(cand_text))

    s1_vec = idf.transform(vectorizer.transform(s1_text))
    cand_vec = idf.transform(vectorizer.transform(cand_text))
    cosine = np.asarray(s1_vec.multiply(cand_vec).sum(axis=1)).ravel()

    # ---- static features ----
    static = [
        build_static_features(
            s1_row["business_name_norm"], s1_row["business_address_norm"], s1_row["country"],
            cand_row["business_name_norm"], cand_row["business_address_norm"], cand_row["country"],
        )
        for (_, s1_row), (_, cand_row) in zip(s1_rows.iterrows(), cand_rows.iterrows())
    ]
    static = np.asarray(static, dtype=np.float32)

    X = np.column_stack([cosine.astype(np.float32), static])

    del s1_vec, cand_vec, s1_text, cand_text
    return X, real_s1_ids, real_cand_ids
