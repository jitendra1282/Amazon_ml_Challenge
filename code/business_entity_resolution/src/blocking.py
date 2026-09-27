"""
Stage 1 — candidate generation ("blocking").

Recall is everything here; Stage 2 (the GBT) is what buys precision back.
Brute-force comparison (S1 x S2+S3) is on the order of trillions of pairs,
so every S1 row must be reduced to a small, high-recall candidate pool
before any expensive similarity work happens.

Three retrieval mechanisms are UNIONed per S1 row (cheap, no ANN/FAISS
needed for this corpus size):

  A. Token-overlap inverted index, ranked by char n-gram TF-IDF cosine
     within the pool (top_k). This is the recall workhorse -- it survives
     word reordering, dropped words, and typos because it doesn't require
     a single leading token to match.
  B. Exact ZIP/PIN match (extracted from the normalized address). Cheap
     and very strong: two records in the same country with the same
     postal code are very likely the same business, even if the name
     text is noisy.
  C. Exact normalized-name match. Also cheap and strong.

All indexes are partitioned by `country` first (never by anything else --
this is what lets the same code handle the unseen France test rows
without modification).

Memory notes (this is what let earlier OOM-crashing versions actually
finish):
  - HashingVectorizer -> fixed feature-space size regardless of corpus
    size or vocabulary (no fit step, no OOV on French n-grams).
  - Inverted-index postings are stored as array('i') (contiguous C ints)
    instead of Python lists -- roughly 9x smaller for the multi-million
    posting lists this corpus produces.
  - S1 rows are blocked in chunks; only the chunk's *union* of candidate
    pools is vectorized at any one time, never the whole S2/S3 corpus at
    once.
"""

from __future__ import annotations

import gc
from array import array as c_array
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from text_utils import content_tokens, extract_zip_pin
import config


def _combined_text(df: pd.DataFrame) -> pd.Series:
    name = df["business_name_norm"].astype(str).fillna("")
    addr = df["business_address_norm"].astype(str).fillna("")
    return (name + " " + addr).str.strip()


def build_vectorizer(ngram_range=config.NGRAM_RANGE, n_features=config.N_FEATURES):
    return HashingVectorizer(
        analyzer="char_wb", ngram_range=ngram_range, n_features=n_features,
        alternate_sign=False, dtype=np.float32, norm=None,
    )


def fit_idf(vectorizer, *frames, sample_cap=config.IDF_SAMPLE_CAP):
    """Fit only the IDF re-weighting on a bounded, per-frame sample of the
    combined corpus (train + test), so rare French n-grams still get a
    sane weight without ever materializing the full corpus at once."""
    per_frame_cap = max(1, sample_cap // max(1, len(frames)))
    parts = []
    for f in frames:
        text = _combined_text(f)
        if len(text) > per_frame_cap:
            text = text.sample(per_frame_cap, random_state=config.RANDOM_SEED)
        parts.append(text)
    sample = pd.concat(parts, ignore_index=True)
    idf = TfidfTransformer()
    idf.fit(vectorizer.transform(sample))
    return idf


def _build_inverted_index(keys_per_row, max_postings):
    """keys_per_row: iterable of (key,) tuples per row index i, already
    including whatever partition prefix (e.g. country) is desired.
    Returns dict[key] -> array('i') of row indices, capped in size."""
    build = defaultdict(list)
    skipped = set()
    for i, keys in enumerate(keys_per_row):
        for key in keys:
            if key in skipped:
                continue
            lst = build[key]
            if len(lst) <= max_postings:
                lst.append(i)
            else:
                skipped.add(key)
    for key in skipped:
        build.pop(key, None)
    index = {key: c_array("i", lst) for key, lst in build.items()}
    del build, skipped
    gc.collect()
    return index


def blocking_candidates(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    vectorizer,
    idf,
    top_k: int = config.TOP_K,
    max_pool_size: int = config.MAX_POOL_SIZE,
    s1_chunk_size: int = config.S1_CHUNK_SIZE,
    candidate_cap: int = config.CANDIDATE_CAP,
):
    """Returns dict[s1_entity_id] -> set(candidate_entity_id)."""

    s1_df = s1_df.reset_index(drop=True)
    s23_df = s23_df.reset_index(drop=True)

    s1_country = s1_df["country"].fillna("").str.strip().str.lower().to_numpy()
    s23_country = s23_df["country"].fillna("").str.strip().str.lower().to_numpy()

    s1_name_norm = s1_df["business_name_norm"].fillna("").to_numpy()
    s23_name_norm = s23_df["business_name_norm"].fillna("").to_numpy()

    s1_tokens = s1_df["business_name_norm"].fillna("").map(content_tokens).to_numpy()
    s23_tokens_series = s23_df["business_name_norm"].fillna("").map(content_tokens)

    s1_zip = s1_df["business_address_norm"].fillna("").map(extract_zip_pin).to_numpy()
    s23_zip = s23_df["business_address_norm"].fillna("").map(extract_zip_pin).to_numpy()

    # ---- Index A: (country, name token) -> token-overlap candidates ----
    token_keys_per_row = (
        [(ctry, t) for t in set(toks)]
        for ctry, toks in zip(s23_country, s23_tokens_series)
    )
    inverted_tokens = _build_inverted_index(token_keys_per_row, config.MAX_POSTINGS_PER_TOKEN)
    del s23_tokens_series
    gc.collect()

    # ---- Index B: (country, zip) -> exact postal-code candidates ----
    zip_keys_per_row = (
        [(ctry, z)] if z else []
        for ctry, z in zip(s23_country, s23_zip)
    )
    inverted_zip = _build_inverted_index(zip_keys_per_row, config.ZIP_EXACT_MAX_PER_TOKEN)

    # ---- Index C: (country, exact normalized name) -> exact-name candidates ----
    name_keys_per_row = (
        [(ctry, n)] if n else []
        for ctry, n in zip(s23_country, s23_name_norm)
    )
    inverted_name = _build_inverted_index(name_keys_per_row, config.EXACT_NAME_MAX_PER_TOKEN)

    s23_text = _combined_text(s23_df)
    s1_vecs = idf.transform(vectorizer.transform(_combined_text(s1_df)))

    n = len(s1_df)
    results = {}

    for start in range(0, n, s1_chunk_size):
        end = min(start + s1_chunk_size, n)

        row_pools = []       # token-overlap pool_counts dict, per row
        row_exact = []       # set of exact-match (zip or name) indices, per row
        union_idx_set = set()

        for i in range(start, end):
            pool_counts = defaultdict(int)
            for t in set(s1_tokens[i]):
                for j in inverted_tokens.get((s1_country[i], t), ()):
                    pool_counts[j] += 1

            exact = set()
            if s1_zip[i]:
                exact.update(inverted_zip.get((s1_country[i], s1_zip[i]), ()))
            if s1_name_norm[i]:
                exact.update(inverted_name.get((s1_country[i], s1_name_norm[i]), ()))

            row_pools.append(pool_counts)
            row_exact.append(exact)
            union_idx_set.update(pool_counts.keys())
            union_idx_set.update(exact)

        if union_idx_set:
            union_idx = np.fromiter(union_idx_set, dtype=np.int32, count=len(union_idx_set))
            local_pos = {g: k for k, g in enumerate(union_idx)}
            pool_vecs_chunk = idf.transform(vectorizer.transform(s23_text.iloc[union_idx]))
        else:
            union_idx = np.array([], dtype=np.int32)
            local_pos = {}
            pool_vecs_chunk = None

        for offset, i in enumerate(range(start, end)):
            eid = s1_df.at[i, "entity_id"]
            pool_counts = row_pools[offset]
            exact = row_exact[offset]

            if not pool_counts and not exact:
                results[eid] = set()
                continue

            # Exact matches (Stage 1C) are always kept -- they're cheap,
            # high-precision signals that should never be dropped just
            # because the token-overlap cosine ranking didn't favor them.
            kept = set(exact)

            # Fill the rest of the budget with the top-cosine token pool.
            pool_idx = list(pool_counts.keys())
            if pool_idx:
                if len(pool_idx) > max_pool_size:
                    pool_idx.sort(key=lambda j: -pool_counts[j])
                    pool_idx = pool_idx[:max_pool_size]
                local_ids = np.array([local_pos[j] for j in pool_idx])
                sims = (s1_vecs[i] @ pool_vecs_chunk[local_ids].T).toarray().ravel()
                k = min(top_k, len(pool_idx))
                top_local = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
                kept.update(pool_idx[p] for p in top_local)

            if len(kept) > candidate_cap:
                # Re-rank everything we're keeping by cosine and truncate,
                # but exact matches are re-added afterwards so they are
                # never the ones trimmed away.
                kept_list = [j for j in kept if j in local_pos]
                local_ids = np.array([local_pos[j] for j in kept_list])
                if len(local_ids):
                    sims_all = (s1_vecs[i] @ pool_vecs_chunk[local_ids].T).toarray().ravel()
                    order = np.argsort(-sims_all)
                    trimmed = [kept_list[o] for o in order[:candidate_cap]]
                else:
                    trimmed = list(kept)[:candidate_cap]
                kept = set(trimmed) | exact

            results[eid] = {s23_df.at[j, "entity_id"] for j in kept}

        del row_pools, row_exact, union_idx_set, union_idx, local_pos, pool_vecs_chunk
        gc.collect()

    return results


def recall_ceiling_report(candidates, ground_truth):
    total_true = total_found = 0
    sizes = []
    for _, row in ground_truth.iterrows():
        s1 = row["source1_entity_id"]
        raw = row["matched_entity_ids"]
        true_ids = set(raw.split(",")) if isinstance(raw, str) and raw.strip() else set()
        cand = candidates.get(s1, set())
        if not true_ids:
            sizes.append(len(cand))
            continue
        total_true += len(true_ids)
        total_found += len(true_ids & cand)
        sizes.append(len(cand))
    return {
        "recall_ceiling": total_found / total_true if total_true else float("nan"),
        "avg_candidate_size": float(np.mean(sizes)) if sizes else 0.0,
        "median_candidate_size": float(np.median(sizes)) if sizes else 0.0,
        "p95_candidate_size": float(np.percentile(sizes, 95)) if sizes else 0.0,
        "zero_candidate_rate": float(np.mean([s == 0 for s in sizes])) if sizes else 0.0,
    }


def write_candidate_pairs_tsv(candidates, s1_entity_ids, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_entity_ids:
            ids = candidates.get(eid, set())
            f.write(f"{eid}\t{','.join(sorted(ids))}\n")


if __name__ == "__main__":
    # Self-test: reordering + dropped word, plus a zip-exact case that the
    # token pool alone would miss if the name text were completely off.
    _s1 = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-4", "S1-5"],
        "business_name_norm": [
            "consulting nyasa nursing private limited",
            "b plus retail incorporated",
            "orelees barbershop",
            "totally different name",
        ],
        "business_address_norm": ["mumbai 400001", "phoenix 85001", "high point", "somewhere 90210"],
        "country": ["India", "US", "US", "US"],
    })
    _s23 = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S2-4", "S3-9"],
        "business_name_norm": [
            "nyasa nursing consulting pvt ltd",
            "b plus retail incorporated",
            "prime money",
            "unrelated business",
        ],
        "business_address_norm": ["mumbai 400001", "tahlequah", "somewhere 90210", "elsewhere"],
        "country": ["India", "US", "US", "US"],
    })
    _vec = build_vectorizer()
    _idf = fit_idf(_vec, _s1, _s23)
    _cands = blocking_candidates(_s1, _s23, _vec, _idf, top_k=5)
    print("Self-test candidates:")
    for k, v in _cands.items():
        print(f"  {k}: {sorted(v)}")
    assert "S2-1" in _cands["S1-1"], "reordered/dropped-word name match failed"
    # S1-5's name text ("totally different name") shares no tokens with
    # S2-4 ("prime money"), so only the exact-ZIP union (90210) can
    # surface it -- this is exactly the Stage 1C case that pure
    # token-overlap blocking would miss.
    assert "S2-4" in _cands["S1-5"], "exact-zip match failed to surface an unrelated-name record"
    print("OK")
