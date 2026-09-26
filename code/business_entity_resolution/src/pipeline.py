"""
Business Entity Resolution Pipeline
====================================
End-to-end pipeline: Data Loading → Blocking → Feature Engineering → Training → Inference

Designed for large-scale data (~2M S1, ~10M S2+S3 entities).
Produces:
  - output/candidate_pairs.tsv   (blocking candidates fed to the model)
  - output/matching_results.tsv  (final predicted matches, scored on leaderboard)

Usage:
    python pipeline.py \
        --train_dir ../../../dataset/train \
        --test_dir  ../../../dataset/test \
        --output_dir ../../../output
"""

import argparse
import gc
import os
import re
import sys
import time
from collections import defaultdict

import jellyfish
import numpy as np
import pandas as pd
from scipy.sparse import vstack as sparse_vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Cleaning & Normalisation
# ─────────────────────────────────────────────────────────────────────────────

# Common business suffixes to normalize
SUFFIX_MAP = {
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bltd\b': 'limited',
    r'\bllc\b': 'limited liability company',
    r'\bpvt\b': 'private',
    r'\bco\b': 'company',
    r'\b&\b': 'and',
    r'\bintl\b': 'international',
    r'\bsvc\b': 'service',
    r'\bsvcs\b': 'services',
    r'\bmfg\b': 'manufacturing',
    r'\bassoc\b': 'associates',
    r'\bgrp\b': 'group',
    r'\bhldg\b': 'holding',
    r'\bhldgs\b': 'holdings',
}

ADDRESS_ABBREV = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bfl\b': 'floor',
    r'\bbldg\b': 'building',
}


def clean_text(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name(text):
    """Normalise business name: clean + expand common suffixes."""
    text = clean_text(text)
    for pat, repl in SUFFIX_MAP.items():
        text = re.sub(pat, repl, text)
    return text.strip()


def normalize_address(text):
    """Normalise address: clean + expand common abbreviations."""
    text = clean_text(text)
    for pat, repl in ADDRESS_ABBREV.items():
        text = re.sub(pat, repl, text)
    return text.strip()


def normalize_country(text):
    """Normalise country label to a canonical lowercase string."""
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_source(path):
    """Load a source TSV and add normalised columns."""
    print(f"  Loading {path} …")
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    df['norm_name'] = df['business_name'].apply(normalize_name)
    df['norm_addr'] = df['business_address'].apply(normalize_address)
    df['norm_country'] = df['country'].apply(normalize_country)
    # Combined text used for TF-IDF blocking
    df['combined'] = df['norm_name'] + ' ' + df['norm_addr']
    print(f"    → {len(df):,} rows")
    return df


def load_ground_truth(path):
    """Load ground truth and return a set of (s1_id, matched_id) pairs."""
    print(f"  Loading ground truth {path} …")
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    gt_pairs = set()
    singleton_s1 = set()
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matched = str(row['matched_entity_ids']).strip()
        if matched:
            for mid in matched.split(','):
                mid = mid.strip()
                if mid:
                    gt_pairs.add((s1_id, mid))
        else:
            singleton_s1.add(s1_id)
    print(f"    → {len(gt_pairs):,} positive pairs, {len(singleton_s1):,} singletons")
    return gt_pairs, singleton_s1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Blocking / Candidate Generation
# ─────────────────────────────────────────────────────────────────────────────

def blocking_tfidf(s1_df, s23_df, top_k=10, batch_size=50000):
    """
    TF-IDF based blocking: for each S1 entity, find top_k most similar
    S2/S3 entities based on cosine similarity of character n-gram TF-IDF vectors.

    Processes S1 entities in batches to manage memory.
    """
    print(f"\n  Fitting TF-IDF vectorizer on {len(s1_df) + len(s23_df):,} texts …")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
        max_features=200000,
        dtype=np.float32,
    )

    # Fit on combined corpus
    all_texts = pd.concat([s1_df['combined'], s23_df['combined']], ignore_index=True)
    vectorizer.fit(all_texts)
    del all_texts
    gc.collect()

    print("  Transforming S2+S3 vectors …")
    s23_vecs = vectorizer.transform(s23_df['combined'])
    s23_ids = s23_df['entity_id'].values

    actual_k = min(top_k, len(s23_df))
    candidates = defaultdict(list)  # s1_id → list of (s23_id, cosine_dist)

    n_batches = (len(s1_df) + batch_size - 1) // batch_size
    print(f"  Computing similarities in {n_batches} batches of {batch_size:,} …")

    for batch_idx in tqdm(range(n_batches), desc="  Blocking batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(s1_df))
        batch_df = s1_df.iloc[start:end]

        s1_vecs = vectorizer.transform(batch_df['combined'])
        sims = cosine_similarity(s1_vecs, s23_vecs)  # (batch, n_s23)

        # Get top-k indices per row
        # Use argpartition for speed (faster than full argsort)
        if actual_k < sims.shape[1]:
            top_indices = np.argpartition(-sims, actual_k, axis=1)[:, :actual_k]
        else:
            top_indices = np.argsort(-sims, axis=1)[:, :actual_k]

        for i in range(len(batch_df)):
            s1_id = batch_df.iloc[i]['entity_id']
            idxs = top_indices[i]
            scores = sims[i, idxs]
            # Sort this small set by score descending
            order = np.argsort(-scores)
            for j in order:
                s23_id = s23_ids[idxs[j]]
                cos_dist = 1.0 - scores[j]
                candidates[s1_id].append((s23_id, float(cos_dist)))

        del s1_vecs, sims, top_indices
        gc.collect()

    del s23_vecs
    gc.collect()

    print(f"  Generated candidates for {len(candidates):,} S1 entities")
    return candidates


def blocking_country_aware(s1_df, s23_df, top_k=10, batch_size=50000):
    """
    Country-aware blocking: block within the same country first, then
    fall back to cross-country blocking for entities with few candidates.
    """
    countries = set(s1_df['norm_country'].unique()) | set(s23_df['norm_country'].unique())
    countries.discard('')

    all_candidates = defaultdict(list)

    for country in sorted(countries):
        s1_country = s1_df[s1_df['norm_country'] == country].copy()
        s23_country = s23_df[s23_df['norm_country'] == country].copy()

        if len(s1_country) == 0 or len(s23_country) == 0:
            continue

        print(f"\n  Blocking for country='{country}': "
              f"{len(s1_country):,} S1 × {len(s23_country):,} S2+S3")

        country_cands = blocking_tfidf(
            s1_country.reset_index(drop=True),
            s23_country.reset_index(drop=True),
            top_k=top_k,
            batch_size=batch_size,
        )

        for s1_id, cand_list in country_cands.items():
            all_candidates[s1_id].extend(cand_list)

    # For S1 entities with no candidates (e.g., empty country), do a global pass
    missing = set(s1_df['entity_id']) - set(all_candidates.keys())
    if missing:
        print(f"\n  {len(missing):,} S1 entities have no country-specific candidates. "
              f"Running global blocking fallback …")
        s1_missing = s1_df[s1_df['entity_id'].isin(missing)].copy()
        fallback = blocking_tfidf(
            s1_missing.reset_index(drop=True),
            s23_df.reset_index(drop=True),
            top_k=top_k,
            batch_size=batch_size,
        )
        for s1_id, cand_list in fallback.items():
            all_candidates[s1_id].extend(cand_list)

    return all_candidates


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_similarity(s1, s2):
    """Token-level Jaccard similarity."""
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def containment_similarity(s1, s2):
    """Fraction of s1 tokens that appear in s2 (asymmetric)."""
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1:
        return 1.0
    return len(set1 & set2) / len(set1)


def sorted_token_jw(s1, s2):
    """Jaro-Winkler on sorted token strings (order-invariant)."""
    t1 = ' '.join(sorted(s1.split()))
    t2 = ' '.join(sorted(s2.split()))
    return jellyfish.jaro_winkler_similarity(t1, t2)


def compute_pair_features(s1_row, s23_row, cosine_dist):
    """Compute feature vector for a single (S1, S2/S3) candidate pair."""
    name1 = s1_row['norm_name']
    name2 = s23_row['norm_name']
    addr1 = s1_row['norm_addr']
    addr2 = s23_row['norm_addr']

    # --- Name features ---
    name_jw = jellyfish.jaro_winkler_similarity(name1, name2)
    name_sorted_jw = sorted_token_jw(name1, name2)
    name_lev = jellyfish.levenshtein_distance(name1, name2)
    name_jaccard = jaccard_similarity(name1, name2)
    name_containment = containment_similarity(name1, name2)
    name_len_diff = abs(len(name1) - len(name2))
    name_len_ratio = min(len(name1), len(name2)) / max(len(name1), len(name2)) if max(len(name1), len(name2)) > 0 else 1.0

    # --- Address features ---
    addr_jw = jellyfish.jaro_winkler_similarity(addr1, addr2)
    addr_sorted_jw = sorted_token_jw(addr1, addr2)
    addr_lev = jellyfish.levenshtein_distance(addr1, addr2)
    addr_jaccard = jaccard_similarity(addr1, addr2)
    addr_containment = containment_similarity(addr1, addr2)
    addr_len_diff = abs(len(addr1) - len(addr2))

    # --- Country features ---
    country_match = 1.0 if s1_row['norm_country'] == s23_row['norm_country'] else 0.0

    # --- Blocking score ---
    cos_dist = cosine_dist

    return [
        cos_dist,
        name_jw, name_sorted_jw, name_lev, name_jaccard, name_containment,
        name_len_diff, name_len_ratio,
        addr_jw, addr_sorted_jw, addr_lev, addr_jaccard, addr_containment,
        addr_len_diff,
        country_match,
    ]


FEATURE_NAMES = [
    'cos_dist',
    'name_jw', 'name_sorted_jw', 'name_lev', 'name_jaccard', 'name_containment',
    'name_len_diff', 'name_len_ratio',
    'addr_jw', 'addr_sorted_jw', 'addr_lev', 'addr_jaccard', 'addr_containment',
    'addr_len_diff',
    'country_match',
]


def build_feature_matrix(candidates_dict, s1_dict, s23_dict, gt_pairs=None, desc="Features"):
    """
    Build feature matrix from candidate dict.
    Returns (DataFrame with features + ids, labels array or None).
    """
    rows = []
    labels = []
    has_gt = gt_pairs is not None

    total_pairs = sum(len(v) for v in candidates_dict.values())
    pbar = tqdm(total=total_pairs, desc=f"  {desc}")

    for s1_id, cand_list in candidates_dict.items():
        s1_row = s1_dict.get(s1_id)
        if s1_row is None:
            pbar.update(len(cand_list))
            continue

        for s23_id, cos_dist in cand_list:
            s23_row = s23_dict.get(s23_id)
            if s23_row is None:
                pbar.update(1)
                continue

            feats = compute_pair_features(s1_row, s23_row, cos_dist)
            rows.append([s1_id, s23_id] + feats)

            if has_gt:
                labels.append(1 if (s1_id, s23_id) in gt_pairs else 0)

            pbar.update(1)

    pbar.close()

    columns = ['source1_entity_id', 'candidate_entity_id'] + FEATURE_NAMES
    df = pd.DataFrame(rows, columns=columns)

    if has_gt:
        return df, np.array(labels, dtype=np.int32)
    return df, None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model Training & Inference
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X, y):
    """Train an XGBoost classifier optimised for F_0.5 (precision-heavy)."""
    # Handle class imbalance: ratio of negatives to positives
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale = n_neg / max(n_pos, 1)
    print(f"  Training set: {len(y):,} pairs, {n_pos:,} positive ({100*n_pos/len(y):.2f}%), "
          f"scale_pos_weight={scale:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        scale_pos_weight=scale,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr',
        tree_method='hist',
    )
    model.fit(X, y, verbose=True)

    # Print feature importances
    importances = model.feature_importances_
    for fname, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"    {fname:25s} {imp:.4f}")

    return model


def predict_matches(model, test_features_df, threshold=0.5):
    """Predict matches using the trained model and a probability threshold."""
    X = test_features_df[FEATURE_NAMES].values
    probs = model.predict_proba(X)[:, 1]
    test_features_df = test_features_df.copy()
    test_features_df['pred_prob'] = probs

    matches = test_features_df[test_features_df['pred_prob'] >= threshold]
    return matches


def find_optimal_threshold(model, val_features_df, val_labels, gt_pairs, singleton_s1):
    """
    Find the threshold that maximises macro-averaged F_0.5 on the validation set.
    """
    X = val_features_df[FEATURE_NAMES].values
    probs = model.predict_proba(X)[:, 1]

    val_df = val_features_df[['source1_entity_id', 'candidate_entity_id']].copy()
    val_df['pred_prob'] = probs

    # Get all S1 IDs (both with and without matches)
    all_s1_ids = set(val_df['source1_entity_id'].unique()) | singleton_s1

    best_threshold = 0.5
    best_f05 = 0.0

    for threshold in np.arange(0.3, 0.85, 0.05):
        preds = val_df[val_df['pred_prob'] >= threshold]
        pred_by_s1 = defaultdict(set)
        for _, row in preds.iterrows():
            pred_by_s1[row['source1_entity_id']].add(row['candidate_entity_id'])

        # Compute macro F_0.5
        f05_scores = []
        for s1_id in all_s1_ids:
            predicted = pred_by_s1.get(s1_id, set())
            true_matches = {m for (s, m) in gt_pairs if s == s1_id}

            if not true_matches and not predicted:
                # Correct singleton
                f05_scores.append(1.0)
            elif not true_matches and predicted:
                # False merge on singleton
                f05_scores.append(0.0)
            elif true_matches and not predicted:
                # Missed all matches
                f05_scores.append(0.0)
            else:
                tp = len(predicted & true_matches)
                fp = len(predicted - true_matches)
                fn = len(true_matches - predicted)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                if precision + recall > 0:
                    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                else:
                    f05 = 0.0
                f05_scores.append(f05)

        macro_f05 = np.mean(f05_scores)
        print(f"    threshold={threshold:.2f}  macro_F0.5={macro_f05:.4f}")
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_threshold = threshold

    print(f"  Best threshold: {best_threshold:.2f} (F_0.5 = {best_f05:.4f})")
    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# 6. Output Formatting
# ─────────────────────────────────────────────────────────────────────────────

def save_candidates(candidates_dict, all_s1_ids, output_path):
    """Save candidate_pairs.tsv with one row per S1 entity."""
    print(f"  Saving {output_path} …")
    with open(output_path, 'w') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in sorted(all_s1_ids):
            cands = candidates_dict.get(s1_id, [])
            # Deduplicate, preserve order
            seen = set()
            unique_ids = []
            for cid, _ in cands:
                if cid not in seen:
                    seen.add(cid)
                    unique_ids.append(cid)
            f.write(f"{s1_id}\t{','.join(unique_ids)}\n")
    print(f"    → written {len(all_s1_ids):,} rows")


def save_matches(matches_df, all_s1_ids, output_path):
    """Save matching_results.tsv with one row per S1 entity."""
    print(f"  Saving {output_path} …")

    # Group predictions by S1 ID
    match_by_s1 = defaultdict(list)
    if matches_df is not None and len(matches_df) > 0:
        for _, row in matches_df.iterrows():
            match_by_s1[row['source1_entity_id']].append(row['candidate_entity_id'])

    with open(output_path, 'w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in sorted(all_s1_ids):
            mids = match_by_s1.get(s1_id, [])
            # Deduplicate
            seen = set()
            unique = []
            for m in mids:
                if m not in seen:
                    seen.add(m)
                    unique.append(m)
            f.write(f"{s1_id}\t{','.join(unique)}\n")
    print(f"    → written {len(all_s1_ids):,} rows")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument('--train_dir', default='../../../dataset/train',
                        help='Path to training data directory')
    parser.add_argument('--test_dir', default='../../../dataset/test',
                        help='Path to test data directory')
    parser.add_argument('--output_dir', default='../../../output',
                        help='Path to output directory')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of candidates per S1 entity from blocking')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override match threshold (auto-tuned by default)')
    parser.add_argument('--batch_size', type=int, default=50000,
                        help='Batch size for TF-IDF similarity computation')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip validation split tuning (use default threshold)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # ── Load data ────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)

    tr_s1 = load_source(os.path.join(args.train_dir, 'train_source1.tsv'))
    tr_s2 = load_source(os.path.join(args.train_dir, 'train_source2.tsv'))
    tr_s3 = load_source(os.path.join(args.train_dir, 'train_source3.tsv'))
    gt_pairs, singleton_s1 = load_ground_truth(
        os.path.join(args.train_dir, 'train_ground_truth.tsv'))

    te_s1 = load_source(os.path.join(args.test_dir, 'test_source1.tsv'))
    te_s2 = load_source(os.path.join(args.test_dir, 'test_source2.tsv'))
    te_s3 = load_source(os.path.join(args.test_dir, 'test_source3.tsv'))

    # Merge S2+S3 for each split
    tr_s23 = pd.concat([tr_s2, tr_s3], ignore_index=True)
    te_s23 = pd.concat([te_s2, te_s3], ignore_index=True)
    del tr_s2, tr_s3, te_s2, te_s3
    gc.collect()

    # Build lookup dicts (entity_id → row dict)
    print("  Building lookup dictionaries …")
    tr_s1_dict = {row['entity_id']: row for _, row in tr_s1.iterrows()}
    tr_s23_dict = {row['entity_id']: row for _, row in tr_s23.iterrows()}
    te_s1_dict = {row['entity_id']: row for _, row in te_s1.iterrows()}
    te_s23_dict = {row['entity_id']: row for _, row in te_s23.iterrows()}

    # ── Blocking ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Blocking / Candidate Generation")
    print("=" * 70)

    print("\n── Training Set Blocking ──")
    tr_candidates = blocking_country_aware(
        tr_s1, tr_s23, top_k=args.top_k, batch_size=args.batch_size)

    print("\n── Test Set Blocking ──")
    te_candidates = blocking_country_aware(
        te_s1, te_s23, top_k=args.top_k, batch_size=args.batch_size)

    # Save candidate_pairs.tsv (test set)
    test_s1_ids = set(te_s1['entity_id'])
    save_candidates(te_candidates, test_s1_ids,
                    os.path.join(args.output_dir, 'candidate_pairs.tsv'))

    # ── Feature Engineering ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Feature Engineering")
    print("=" * 70)

    print("\n── Training Features ──")
    tr_feat_df, tr_labels = build_feature_matrix(
        tr_candidates, tr_s1_dict, tr_s23_dict, gt_pairs=gt_pairs, desc="Train features")

    print("\n── Test Features ──")
    te_feat_df, _ = build_feature_matrix(
        te_candidates, te_s1_dict, te_s23_dict, gt_pairs=None, desc="Test features")

    # ── Model Training ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Model Training")
    print("=" * 70)

    X_train = tr_feat_df[FEATURE_NAMES].values
    model = train_model(X_train, tr_labels)

    # ── Threshold Tuning ─────────────────────────────────────────────────
    if args.threshold is not None:
        threshold = args.threshold
        print(f"\n  Using user-specified threshold: {threshold:.2f}")
    elif not args.skip_validation:
        print("\n  Tuning threshold on training set (in-sample — for a better estimate, "
              "use a held-out validation split) …")
        threshold = find_optimal_threshold(
            model, tr_feat_df, tr_labels, gt_pairs, singleton_s1)
    else:
        threshold = 0.6
        print(f"\n  Using default threshold: {threshold:.2f}")

    # ── Inference ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Inference")
    print("=" * 70)

    matches_df = predict_matches(model, te_feat_df, threshold=threshold)
    print(f"  Predicted {len(matches_df):,} positive pairs "
          f"for {matches_df['source1_entity_id'].nunique():,} S1 entities")

    # Save matching_results.tsv
    save_matches(matches_df, test_s1_ids,
                 os.path.join(args.output_dir, 'matching_results.tsv'))

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Pipeline completed in {elapsed/60:.1f} minutes.")
    print(f"Output files saved to: {args.output_dir}/")
    print(f"  - candidate_pairs.tsv")
    print(f"  - matching_results.tsv")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
