"""
Stage 2 training.

Reads output/train_candidate_pairs.tsv (Stage 1 run on the TRAIN sources)
and train_ground_truth.tsv, labels every candidate pair, splits by S1
ENTITY (never by raw row -- a row-level split would leak candidates of
the same entity across train/validation and give an unrealistically
rosy validation score), trains an XGBoost classifier, and tunes the
decision threshold directly against validation F0.5 (the actual
competition metric) rather than using a fixed 0.5 cutoff.
"""

from __future__ import annotations

import csv
import gc
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from features import create_feature_batch


def _load_ground_truth(path: str) -> dict:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    truth_map = {}
    for _, row in gt.iterrows():
        raw = row["matched_entity_ids"]
        truth_map[row["source1_entity_id"]] = (
            set(x.strip() for x in raw.split(",") if x.strip()) if raw.strip() else set()
        )
    return truth_map


def _iter_candidate_pairs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            raw = row["candidate_entity_ids"]
            if not raw.strip():
                continue
            for cand_id in raw.split(","):
                cand_id = cand_id.strip()
                if cand_id:
                    yield s1_id, cand_id


def build_training_matrix(train_s1, train_s23, vectorizer, idf,
                           candidate_path=config.TRAIN_CANDIDATE_PATH,
                           ground_truth_path=config.GROUND_TRUTH_PATH,
                           pair_batch_size=config.PAIR_BATCH_SIZE):
    print("Building labeled training pairs...")
    truth_map = _load_ground_truth(ground_truth_path)
    print(f"  ground-truth S1 entities: {len(truth_map):,}")

    s1_index = pd.Index(train_s1["entity_id"])
    s23_index = pd.Index(train_s23["entity_id"])

    X_chunks, y_chunks, s1_for_rows = [], [], []
    pair_s1, pair_cand = [], []
    total_pairs = 0

    def _flush():
        nonlocal pair_s1, pair_cand, total_pairs
        if not pair_s1:
            return
        X_batch, valid_s1, valid_cand = create_feature_batch(
            pair_s1, pair_cand, train_s1, train_s23, s1_index, s23_index, vectorizer, idf
        )
        if len(X_batch):
            y_batch = np.asarray(
                [float(cid in truth_map.get(sid, set())) for sid, cid in zip(valid_s1, valid_cand)],
                dtype=np.float32,
            )
            X_chunks.append(X_batch)
            y_chunks.append(y_batch)
            s1_for_rows.extend(valid_s1)
        total_pairs += len(pair_s1)
        if total_pairs % 200_000 < pair_batch_size:
            print(f"  training pairs processed: {total_pairs:,}")
        pair_s1, pair_cand = [], []
        gc.collect()

    for s1_id, cand_id in _iter_candidate_pairs(candidate_path):
        pair_s1.append(s1_id)
        pair_cand.append(cand_id)
        if len(pair_s1) >= pair_batch_size:
            _flush()
    _flush()

    X = np.vstack(X_chunks).astype(np.float32)
    y = np.concatenate(y_chunks).astype(np.float32)
    del X_chunks, y_chunks
    gc.collect()

    print(f"Training feature matrix: {X.shape}, positive={int(y.sum()):,}, negative={int((y == 0).sum()):,}")
    return X, y, s1_for_rows


def entity_level_split(X, y, s1_for_rows, valid_fraction=config.VALIDATION_FRACTION, seed=config.RANDOM_SEED):
    unique_s1 = np.array(list(set(s1_for_rows)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_s1)
    n_valid = int(len(unique_s1) * valid_fraction)
    valid_set = set(unique_s1[:n_valid])
    is_valid = np.array([sid in valid_set for sid in s1_for_rows])
    return X[~is_valid], y[~is_valid], X[is_valid], y[is_valid], valid_set


def f_beta_score(y_true, y_prob, threshold, beta=config.F_BETA):
    pred = (y_prob >= threshold).astype(np.int8)
    tp = np.sum((y_true == 1) & (pred == 1))
    fp = np.sum((y_true == 0) & (pred == 1))
    fn = np.sum((y_true == 1) & (pred == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def tune_threshold(y_valid, valid_prob, grid=config.THRESHOLD_GRID):
    scores = [f_beta_score(y_valid, valid_prob, t) for t in grid]
    best_idx = int(np.argmax(scores))
    return float(grid[best_idx]), float(scores[best_idx])


def train_model(X_train, y_train, X_valid, y_valid):
    positive = max(1, int(y_train.sum()))
    negative = max(1, int((y_train == 0).sum()))
    scale_pos_weight = negative / positive
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(**config.XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    print(f"GBT training time: {time.time() - t0:.1f}s")
    return model


def run(train_s1, train_s23, vectorizer, idf):
    X, y, s1_for_rows = build_training_matrix(train_s1, train_s23, vectorizer, idf)
    X_train, y_train, X_valid, y_valid, valid_s1_set = entity_level_split(X, y, s1_for_rows)
    print(f"train rows: {len(y_train):,} | valid rows: {len(y_valid):,} | valid S1 entities: {len(valid_s1_set):,}")

    model = train_model(X_train, y_train, X_valid, y_valid)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    best_threshold, best_f05 = tune_threshold(y_valid, valid_prob)
    print(f"Best validation threshold: {best_threshold:.2f} | Validation F{config.F_BETA}: {best_f05:.5f}")

    model.save_model(config.MODEL_PATH)
    with open(config.THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": best_threshold, "f_beta_score": best_f05,
                   "beta": config.F_BETA, "feature_names": config.FEATURE_NAMES}, f, indent=2)

    del X, y, X_train, y_train, X_valid, y_valid, valid_prob
    gc.collect()
    return model, best_threshold, best_f05
