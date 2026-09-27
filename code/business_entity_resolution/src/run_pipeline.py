"""
Business Entity Resolution — full pipeline entry point.

    python run_pipeline.py                     # run everything
    python run_pipeline.py --skip-blocking      # reuse existing candidate_pairs.tsv files
    python run_pipeline.py --skip-training       # reuse the cached Stage 2 model + threshold
    python run_pipeline.py --top-k 60            # override Stage 1 pool size for this run

Every expensive step checks for its output/cache first and skips itself
if it's already there (unless --force is passed), so a crash 80% of the
way through inference doesn't cost you the earlier hours of blocking and
training.

Memory profile targeted: full pipeline under ~25 GB RAM on a ~1 GB raw
dataset. This comes from (a) a HashingVectorizer with a fixed feature
space instead of a fitted vocabulary, (b) array('i') inverted-index
postings instead of Python lists, (c) chunked/batched processing at every
stage with `gc.collect()` between chunks, and (d) never holding more than
one side (S1 chunk x its candidate pool) of the giant cross product in
memory at once.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import data_io
import cache_utils
from blocking import build_vectorizer, fit_idf, blocking_candidates, recall_ceiling_report, write_candidate_pairs_tsv
import train_stage2
import predict_stage2
import xgboost as xgb


def _section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_or_build_vectorizer(train_s1, train_s23, test_s1, test_s23, force=False):
    if not force:
        cached = cache_utils.load_vectorizer()
        if cached is not None:
            print("  [cache hit] vectorizer + IDF")
            return cached
    vectorizer = build_vectorizer()
    idf = fit_idf(vectorizer, train_s1, train_s23, test_s1, test_s23)
    cache_utils.save_vectorizer(vectorizer, idf)
    print(f"  vectorizer ready (n_features={vectorizer.n_features:,}), IDF fit and cached")
    return vectorizer, idf


def run_blocking_stage(s1_df, s23_df, vectorizer, idf, out_path, top_k,
                        ground_truth=None, force=False, skip=False):
    if skip:
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"--skip-blocking was given but {out_path} does not exist")
        print(f"  [skip] reusing existing {out_path}")
        return
    if not force and os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists (use --force to recompute)")
        return
    t0 = time.time()
    candidates = blocking_candidates(s1_df, s23_df, vectorizer, idf, top_k=top_k)
    write_candidate_pairs_tsv(candidates, s1_df["entity_id"], out_path)
    print(f"  blocking wrote {out_path} in {time.time() - t0:.1f}s")
    if ground_truth is not None:
        report = recall_ceiling_report(candidates, ground_truth)
        print(f"  recall ceiling report: {report}")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument("--skip-blocking", action="store_true", help="reuse existing candidate_pairs.tsv files")
    parser.add_argument("--skip-training", action="store_true", help="reuse cached Stage 2 model + threshold")
    parser.add_argument("--no-cache", action="store_true", help="ignore normalization/vectorizer caches")
    parser.add_argument("--force", action="store_true", help="recompute everything even if outputs already exist")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    use_cache = not args.no_cache
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    _section("[1/6] Loading + normalizing TRAIN data")
    train_s1, train_s2, train_s3, ground_truth = data_io.load_all_train(use_cache)
    train_s23 = data_io.concat_s23(train_s2, train_s3)

    _section("[2/6] Loading + normalizing TEST data")
    test_s1, test_s2, test_s3 = data_io.load_all_test(use_cache)
    test_s23 = data_io.concat_s23(test_s2, test_s3)

    _section("[3/6] Vectorizer + IDF (shared by Stage 1 and Stage 2)")
    vectorizer, idf = get_or_build_vectorizer(train_s1, train_s23, test_s1, test_s23, force=args.force)

    _section("[4/6] Stage 1 — blocking (candidate generation)")
    print("-- TRAIN candidates (for labeling + Stage 2 training) --")
    run_blocking_stage(train_s1, train_s23, vectorizer, idf, config.TRAIN_CANDIDATE_PATH,
                        args.top_k, ground_truth=ground_truth, force=args.force, skip=args.skip_blocking)
    print("-- TEST candidates (for final scoring) --")
    run_blocking_stage(test_s1, test_s23, vectorizer, idf, config.TEST_CANDIDATE_PATH,
                        args.top_k, ground_truth=None, force=args.force, skip=args.skip_blocking)

    _section("[5/6] Stage 2 — training the GBT matcher")
    if not args.force and args.skip_training and os.path.exists(config.MODEL_PATH) and os.path.exists(config.THRESHOLD_PATH):
        print("  [skip] loading cached model + threshold")
        model = xgb.XGBClassifier()
        model.load_model(config.MODEL_PATH)
        import json
        with open(config.THRESHOLD_PATH) as f:
            meta = json.load(f)
        threshold = meta["threshold"]
        print(f"  loaded threshold={threshold:.2f}, cached F{meta['beta']}={meta['f_beta_score']:.5f}")
    else:
        model, threshold, best_f05 = train_stage2.run(train_s1, train_s23, vectorizer, idf)

    _section("[6/6] Stage 2 — scoring TEST candidates -> matching_results.tsv")
    stats = predict_stage2.score_test_candidates(model, threshold, test_s1, test_s23, vectorizer, idf)

    _section("PIPELINE COMPLETE")
    print(f"Output: {config.MATCHING_RESULTS_PATH}")
    print(f"Candidate file: {config.TEST_CANDIDATE_PATH}")
    print(stats)


if __name__ == "__main__":
    main()
