"""
Smoke test: Run the pipeline on a tiny subset (100 S1 rows) to verify
the code works end-to-end before committing to a full run.
"""
import pandas as pd
import os, sys, tempfile, shutil

# Paths dynamically resolved relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../dataset/train'))
TEST_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../dataset/test'))

print("=== Smoke Test: Loading tiny subsets ===")

# Load just the first 100 rows of each
tr_s1 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t', dtype=str, nrows=100, keep_default_na=False)
tr_s2 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t', dtype=str, nrows=500, keep_default_na=False)
tr_s3 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t', dtype=str, nrows=500, keep_default_na=False)
gt    = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t', dtype=str, nrows=100, keep_default_na=False)

te_s1 = pd.read_csv(os.path.join(TEST_DIR, 'test_source1.tsv'), sep='\t', dtype=str, nrows=100, keep_default_na=False)
te_s2 = pd.read_csv(os.path.join(TEST_DIR, 'test_source2.tsv'), sep='\t', dtype=str, nrows=500, keep_default_na=False)
te_s3 = pd.read_csv(os.path.join(TEST_DIR, 'test_source3.tsv'), sep='\t', dtype=str, nrows=500, keep_default_na=False)

print(f"Train S1: {len(tr_s1)}, S2: {len(tr_s2)}, S3: {len(tr_s3)}, GT: {len(gt)}")
print(f"Test  S1: {len(te_s1)}, S2: {len(te_s2)}, S3: {len(te_s3)}")
print(f"\nColumns: {list(tr_s1.columns)}")
print(f"GT Columns: {list(gt.columns)}")
print(f"\nSample S1 row:\n{tr_s1.iloc[0].to_dict()}")
print(f"\nSample GT row:\n{gt.iloc[0].to_dict()}")

# Write tiny subsets to temp dir and run the pipeline
tmp_train = tempfile.mkdtemp(prefix='er_train_', dir='.')
tmp_test  = tempfile.mkdtemp(prefix='er_test_', dir='.')
tmp_out   = tempfile.mkdtemp(prefix='er_out_', dir='.')

try:
    tr_s1.to_csv(os.path.join(tmp_train, 'train_source1.tsv'), sep='\t', index=False)
    tr_s2.to_csv(os.path.join(tmp_train, 'train_source2.tsv'), sep='\t', index=False)
    tr_s3.to_csv(os.path.join(tmp_train, 'train_source3.tsv'), sep='\t', index=False)
    gt.to_csv(os.path.join(tmp_train, 'train_ground_truth.tsv'), sep='\t', index=False)
    
    te_s1.to_csv(os.path.join(tmp_test, 'test_source1.tsv'), sep='\t', index=False)
    te_s2.to_csv(os.path.join(tmp_test, 'test_source2.tsv'), sep='\t', index=False)
    te_s3.to_csv(os.path.join(tmp_test, 'test_source3.tsv'), sep='\t', index=False)
    
    print(f"\nTemp dirs created: train={tmp_train}, test={tmp_test}, out={tmp_out}")
    
    # Import and run pipeline
    sys.path.insert(0, '.')
    import pipeline
    
    sys.argv = ['pipeline.py',
                '--train_dir', tmp_train,
                '--test_dir', tmp_test,
                '--output_dir', tmp_out,
                '--top_k', '5',
                '--batch_size', '50',
                '--skip_validation']
    
    pipeline.main()
    
    # Verify outputs
    cand_path = os.path.join(tmp_out, 'candidate_pairs.tsv')
    match_path = os.path.join(tmp_out, 'matching_results.tsv')
    
    assert os.path.exists(cand_path), f"candidate_pairs.tsv not created!"
    assert os.path.exists(match_path), f"matching_results.tsv not created!"
    
    cand_df = pd.read_csv(cand_path, sep='\t', dtype=str)
    match_df = pd.read_csv(match_path, sep='\t', dtype=str)
    
    print(f"\n=== Smoke Test Results ===")
    print(f"candidate_pairs.tsv: {len(cand_df)} rows, cols={list(cand_df.columns)}")
    print(f"matching_results.tsv: {len(match_df)} rows, cols={list(match_df.columns)}")
    
    # Check every test S1 ID has a row
    test_s1_ids = set(te_s1['entity_id'])
    cand_s1_ids = set(cand_df['source1_entity_id'])
    match_s1_ids = set(match_df['source1_entity_id'])
    
    assert test_s1_ids == cand_s1_ids, f"Missing S1 IDs in candidates: {test_s1_ids - cand_s1_ids}"
    assert test_s1_ids == match_s1_ids, f"Missing S1 IDs in matches: {test_s1_ids - match_s1_ids}"
    
    print(f"All {len(test_s1_ids)} test S1 IDs present in both output files ✓")
    print(f"\nSample candidate row:\n{cand_df.iloc[0].to_dict()}")
    print(f"\nSample matching row:\n{match_df.iloc[0].to_dict()}")
    print(f"\n=== SMOKE TEST PASSED ✓ ===")
    
finally:
    shutil.rmtree(tmp_train, ignore_errors=True)
    shutil.rmtree(tmp_test, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)
