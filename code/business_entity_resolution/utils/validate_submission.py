#!/usr/bin/env python3
"""
Validate matching_results.tsv and candidate_pairs.tsv against the
competition's format rules, entirely offline and stdlib-only, so a
rejection is caught locally instead of costing a submission.

Usage:
    python3 utils/validate_submission.py \\
        --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv \\
        --test-dir dataset/test

Prints PASS (exit 0) or a numbered list of issues (exit 1). It only reads
the two output files and the test source files -- it does not compute
your F0.5 score.
"""

import argparse
import csv
import os
import sys


def read_ids(path, prefix):
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            eid = row[0]
            if eid.startswith(prefix):
                ids.add(eid)
    return ids


def read_result_file(path, id_col, list_col_name):
    """Returns list of (row_number, source1_entity_id, [ids...]) plus the header."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for i, row in enumerate(reader, start=2):  # 1-indexed + header row
            if len(row) == 0 or (len(row) == 1 and row[0].strip() == ""):
                continue
            if len(row) < 2:
                sid = row[0] if row else ""
                ids = []
            else:
                sid = row[0]
                raw = row[1]
                ids = [x.strip() for x in raw.split(",") if x.strip()] if raw.strip() else []
            rows.append((i, sid, ids))
    return header, rows


def validate(matching_path, candidate_path, test_dir):
    issues = []

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for p in (s1_path, s2_path, s3_path, matching_path, candidate_path):
        if not os.path.exists(p):
            issues.append(f"Missing required file: {p}")
    if issues:
        return issues

    test_s1_ids = read_ids(s1_path, "S1-")
    test_s23_ids = read_ids(s2_path, "S2-") | read_ids(s3_path, "S3-")

    m_header, m_rows = read_result_file(matching_path, "source1_entity_id", "matched_entity_ids")
    c_header, c_rows = read_result_file(candidate_path, "source1_entity_id", "candidate_entity_ids")

    expected_header = ["source1_entity_id", "matched_entity_ids"]
    if m_header != expected_header:
        issues.append(f"matching_results.tsv header is {m_header}, expected {expected_header}")

    expected_c_header = ["source1_entity_id", "candidate_entity_ids"]
    if c_header != expected_c_header:
        issues.append(f"candidate_pairs.tsv header is {c_header}, expected {expected_c_header}")

    def check_rows(rows, label):
        seen = set()
        s1_to_ids = {}
        dup_s1 = set()
        for lineno, sid, ids in rows:
            if sid in seen:
                dup_s1.add(sid)
            seen.add(sid)

            if not sid.startswith("S1-"):
                issues.append(f"{label} line {lineno}: source1_entity_id {sid!r} is not an S1- id")
            elif sid not in test_s1_ids:
                issues.append(f"{label} line {lineno}: {sid} is not in the test set")

            seen_ids_this_row = set()
            for cid in ids:
                if cid in seen_ids_this_row:
                    issues.append(f"{label} line {lineno} ({sid}): duplicate id {cid} within the row")
                seen_ids_this_row.add(cid)

                if cid.startswith("S1-"):
                    issues.append(f"{label} line {lineno} ({sid}): {cid} is a Source 1 id (self-match not allowed)")
                elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                    issues.append(f"{label} line {lineno} ({sid}): {cid} has an unrecognized prefix")
                elif cid not in test_s23_ids:
                    issues.append(f"{label} line {lineno} ({sid}): {cid} does not exist in the test set")

            s1_to_ids[sid] = set(ids)

        for sid in dup_s1:
            issues.append(f"{label}: duplicate source1_entity_id row for {sid}")

        missing = test_s1_ids - seen
        if missing:
            sample = ", ".join(sorted(missing)[:10])
            issues.append(f"{label}: {len(missing):,} test S1 entities missing from this file (e.g. {sample})")

        return s1_to_ids

    matched_map = check_rows(m_rows, "matching_results.tsv")
    candidate_map = check_rows(c_rows, "candidate_pairs.tsv")

    for sid, matched_ids in matched_map.items():
        cand_ids = candidate_map.get(sid, set())
        stray = matched_ids - cand_ids
        if stray:
            sample = ", ".join(sorted(stray)[:10])
            issues.append(
                f"matching_results.tsv ({sid}): {len(stray)} matched id(s) never appeared as a candidate "
                f"(e.g. {sample}) -- likely a pipeline bug"
            )

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)

    if not issues:
        print("PASS")
        sys.exit(0)
    else:
        print(f"{len(issues)} issue(s) found:\n")
        for i, issue in enumerate(issues, start=1):
            print(f"{i}. {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
