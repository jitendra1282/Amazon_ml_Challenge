"""
Central configuration for the Business Entity Resolution pipeline.

Every tunable knob lives here so a re-run with different settings
(e.g. a bigger top_k, a smaller pool cap for a memory-constrained box)
never requires touching the actual pipeline logic.
"""

import os

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

TRAIN_S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
GROUND_TRUTH_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1_PATH = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2_PATH = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3_PATH = os.path.join(TEST_DIR, "test_source3.tsv")

TRAIN_CANDIDATE_PATH = os.path.join(OUTPUT_DIR, "train_candidate_pairs.tsv")
TEST_CANDIDATE_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")

VECTORIZER_CACHE_PATH = os.path.join(CACHE_DIR, "vectorizer_idf.pkl")
MODEL_PATH = os.path.join(CACHE_DIR, "stage2_xgb.json")
THRESHOLD_PATH = os.path.join(CACHE_DIR, "stage2_threshold.json")
NORMALIZED_CACHE_DIR = os.path.join(CACHE_DIR, "normalized")

# ------------------------------------------------------------------
# Stage 1 — blocking / candidate generation
# ------------------------------------------------------------------

# Char n-gram TF-IDF (hashing, so memory is fixed regardless of corpus size).
NGRAM_RANGE = (2, 4)
N_FEATURES = 2 ** 18          # 262,144 dims, float32 -> fixed & bounded memory
IDF_SAMPLE_CAP = 500_000      # rows sampled to fit the IDF re-weighting

# Bumped from 20 -> 45 per the review: bigger pools are "free" upside now
# that Stage 2 is a real classifier doing the precision work, not a
# hand-tuned threshold.
TOP_K = 45

# Safety valves so one pathological S1 row (a one-word generic name that
# matches half the corpus) can't blow up memory or runtime.
MAX_POOL_SIZE = 6000            # cosine is computed over at most this many candidates
MAX_POSTINGS_PER_TOKEN = 20_000  # inverted-index list cap per (country, token)
S1_CHUNK_SIZE = 5_000            # how many S1 rows are blocked together per chunk

# Exact-field retrieval (Stage 1C in the review): cheap, high-precision
# signals that are always unioned into the candidate pool regardless of
# what the cosine-ranked token pool contains.
ZIP_EXACT_MAX_PER_TOKEN = 20_000   # cap on (country, zip) postings list
EXACT_NAME_MAX_PER_TOKEN = 20_000  # cap on (country, exact-normalized-name) postings list

# Final candidate cap per S1 entity (Stage 1E): exact-field matches are
# always kept; the remaining budget is filled by top-cosine candidates.
CANDIDATE_CAP = 150

# ------------------------------------------------------------------
# Stage 2 — GBT matcher
# ------------------------------------------------------------------

PAIR_BATCH_SIZE = 20_000   # candidate pairs featurized together
S1_WRITE_BATCH_SIZE = 5_000  # S1 rows buffered before a scoring+write pass

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.9,
    min_child_weight=2,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=4,
    random_state=RANDOM_SEED,
)

# F-beta used by the competition (precision weighted 2x over recall).
F_BETA = 0.5

THRESHOLD_GRID = [round(x, 3) for x in
                  __import__("numpy").arange(0.10, 0.991, 0.01)]

FEATURE_NAMES = [
    "cosine_sim",
    "name_jaccard",
    "address_jaccard",
    "zip_exact",
    "country_exact",
    "name_exact",
    "name_fuzzy_ratio",
    "address_fuzzy_ratio",
    "name_missing",
    "address_missing",
    "zip_missing_both",
]

KEEP_COLS = [
    "entity_id",
    "country",
    "business_name_norm",
    "business_address_norm",
]

# Rough size of the full test S1 file, used only to print an ETA during
# the long inference loop. Harmless if it's off; update if your test set
# size differs.
EXPECTED_TEST_S1_ROWS = 1_732_544
