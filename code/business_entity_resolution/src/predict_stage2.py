"""
Score the TEST candidate_pairs.tsv with the trained Stage 2 model and
write matching_results.tsv. Streams the candidate file rather than
loading it whole, and frees each batch's arrays before starting the
next -- this is what keeps peak memory bounded regardless of how many
millions of S1 rows the test set has.
"""

from __future__ import annotations

import csv
import gc
import time

import numpy as np
import pandas as pd

import config
from features import create_feature_batch


def score_test_candidates(model, threshold, test_s1, test_s23, vectorizer, idf,
                           candidate_path=config.TEST_CANDIDATE_PATH,
                           output_path=config.MATCHING_RESULTS_PATH,
                           s1_batch_size=config.S1_WRITE_BATCH_SIZE,
                           pair_batch_size=config.PAIR_BATCH_SIZE,
                           expected_rows=config.EXPECTED_TEST_S1_ROWS):

    s1_index = pd.Index(test_s1["entity_id"])
    s23_index = pd.Index(test_s23["entity_id"])

    def score_batch(s1_ids, cand_ids):
        results = {}
        for start in range(0, len(s1_ids), pair_batch_size):
            end = min(start + pair_batch_size, len(s1_ids))
            X_batch, valid_s1, valid_cand = create_feature_batch(
                s1_ids[start:end], cand_ids[start:end], test_s1, test_s23, s1_index, s23_index, vectorizer, idf
            )
            if len(X_batch) == 0:
                continue
            probs = model.predict_proba(X_batch)[:, 1]
            for sid, cid, p in zip(valid_s1, valid_cand, probs):
                if p >= threshold:
                    results.setdefault(sid, set()).add(cid)
            del X_batch
        return results

    total_s1 = total_pairs = total_matches = 0
    start_time = time.time()
    batch_s1_ids, batch_cand_lists = [], []

    with open(candidate_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        writer = csv.writer(f_out, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        def flush():
            nonlocal total_s1, total_pairs, total_matches, batch_s1_ids, batch_cand_lists
            if not batch_s1_ids:
                return
            pair_s1, pair_cand = [], []
            for sid, cids in zip(batch_s1_ids, batch_cand_lists):
                for cid in cids:
                    pair_s1.append(sid)
                    pair_cand.append(cid)

            batch_results = score_batch(pair_s1, pair_cand)

            for sid in batch_s1_ids:
                matches = batch_results.get(sid, set())
                writer.writerow([sid, ",".join(sorted(matches))])
                if matches:
                    total_matches += len(matches)

            total_s1 += len(batch_s1_ids)
            elapsed = time.time() - start_time
            rate = total_s1 / elapsed if elapsed > 0 else 0
            remaining = max(0, expected_rows - total_s1)
            eta_min = (remaining / rate / 60) if rate > 0 else 0
            print(f"Processed {total_s1:,}/{expected_rows:,} ({total_s1/expected_rows:.1%}) | "
                  f"pairs: {total_pairs:,} | matches: {total_matches:,} | ETA: {eta_min:.1f} min")

            batch_s1_ids, batch_cand_lists = [], []
            del pair_s1, pair_cand, batch_results
            gc.collect()

        for row in reader:
            sid = row["source1_entity_id"]
            raw = row["candidate_entity_ids"]
            cids = [x.strip() for x in raw.split(",") if x.strip()] if raw.strip() else []
            batch_s1_ids.append(sid)
            batch_cand_lists.append(cids)
            total_pairs += len(cids)
            if len(batch_s1_ids) >= s1_batch_size:
                flush()
        flush()

    print(f"Wrote {output_path}: {total_s1:,} S1 rows, {total_matches:,} matched pairs")
    return {"total_s1": total_s1, "total_pairs": total_pairs, "total_matches": total_matches}
