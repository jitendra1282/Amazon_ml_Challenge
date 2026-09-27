# Business Entity Resolution — Stage 1 (Blocking) + Stage 2 (GBT) Pipeline

A single, resumable pipeline that takes the 3-source TSVs and ground truth
and produces `output/matching_results.tsv` (scored) and
`output/candidate_pairs.tsv` (blocking output, for recall/audit).

Model: XGBoost (`binary:logistic`), open-source (Apache 2.0), a few
hundred KB on disk — comfortably inside the "MIT/Apache-2.0, ≤8B params"
constraint. No external data, APIs, or lookups are used anywhere in the
pipeline (fully compliant with the fair-play rules).

## Architecture

```
TRAIN/TEST source1/2/3 (tsv)
        │
        ▼
Stage 0 — normalization.py
  (suffix map, address abbreviations, landmark extraction; never
   branches on country -> the same code handles unseen France rows)
        │
        ▼
Stage 1 — blocking.py            <- optimizes RECALL
  A. token-overlap inverted index, ranked by char n-gram TF-IDF cosine
     (top_k=45 candidates per S1 row)
  B. exact ZIP/PIN match           } unioned in regardless of
  C. exact normalized-name match   } the cosine ranking (cheap, high-precision)
  -> candidate_pairs.tsv (both train and test)
        │
        ▼
Stage 2 — features.py + train_stage2.py / predict_stage2.py   <- optimizes PRECISION
  11 features per (S1, candidate) pair: TF-IDF cosine, name/address
  token Jaccard, exact ZIP/country/name match, RapidFuzz name/address
  ratio, and 3 missingness flags.
  XGBoost classifier, trained on TRAIN candidate pairs labeled against
  train_ground_truth.tsv, split by S1 ENTITY (not by row) to avoid leakage.
  Decision threshold tuned against validation F0.5 (the real competition
  metric), not a fixed 0.5 cutoff.
        │
        ▼
matching_results.tsv
```

## Why this design

- **Stage 1 optimizes recall, Stage 2 optimizes precision.** Blocking
  never tries to be smart about *which* candidate is the right one — it
  just needs the true match to survive into the candidate pool. F0.5 is
  precision-heavy, so that's exactly the GBT's job in Stage 2.
- **No ANN/FAISS/embeddings.** An earlier test with a multilingual
  sentence-embedding model (multilingual-e5) could not separate
  "Smith Consulting LLC" from "Smith Consulting Group Inc" (0.975 cosine,
  indistinguishable from a true match), so embeddings were dropped rather
  than layered in as a fourth retriever. The char n-gram TF-IDF signal
  already used for ranking is repurposed as a Stage 2 feature instead.
- **Exact-field retrieval (ZIP/PIN, exact name) is unioned into the
  candidate pool unconditionally.** These are cheap and often catch the
  true match when the token-overlap ranking would have missed it (e.g. a
  transliterated name with a completely different token set, but the
  same postal code).
- **`top_k` raised from 20 to 45.** With a real GBT doing the precision
  work in Stage 2, a wider Stage 1 pool is close to free upside on recall.
- **Vectorizer/IDF and normalized frames are cached to disk**
  (`cache/`), so re-running Stage 2 (or resuming after a crash) never
  redoes Stage 0/1 work from scratch.
- **Everything is chunked and streamed**: HashingVectorizer gives a fixed
  feature-space size regardless of corpus size (no OOV on French
  n-grams, no vocabulary to hold in memory); inverted-index postings are
  `array('i')`, not Python lists; candidate pairs are read and scored in
  bounded batches with `gc.collect()` between them. This is what keeps
  the whole pipeline under ~25 GB RAM.

## Setup

```bash
pip install -r requirements.txt
```

Place the competition data under:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

## Run

```bash
cd src
python run_pipeline.py
```

This runs, in order: normalize (cached) -> build vectorizer/IDF (cached)
-> block TRAIN + TEST -> train Stage 2 GBT + tune threshold -> score TEST
candidates -> write `output/matching_results.tsv`.

Useful flags:

```bash
# Resume after a crash: reuse existing candidate_pairs.tsv files
python run_pipeline.py --skip-blocking

# Resume after a crash: reuse the cached model + threshold, just re-score
python run_pipeline.py --skip-blocking --skip-training

# Try a bigger/smaller candidate pool without touching the code
python run_pipeline.py --top-k 60

# Ignore all caches and recompute everything from scratch
python run_pipeline.py --force --no-cache
```

Each individual module also has a quick self-test:

```bash
python normalization.py   # known tricky name pairs
python blocking.py        # tiny synthetic dataset: reordering, dropped
                           # words, and an exact-ZIP-match case
```

## Validate before submitting

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Measuring your own score

The pipeline prints a validation F0.5 score during Stage 2 training
(entity-level split of the training data), and a recall-ceiling report
for Stage 1 blocking (candidate recall against `train_ground_truth.tsv`).
Both are logged to stdout — pipe `run_pipeline.py`'s output to a file if
you want to keep a record across runs while you tune `--top-k` or the
XGBoost hyperparameters in `config.py`.

## Tuning knobs (all in `config.py`)

| Knob | Default | Effect |
|---|---|---|
| `TOP_K` | 45 | Stage 1 pool size per S1 row (token-overlap branch) |
| `CANDIDATE_CAP` | 150 | hard cap on total candidates per S1 row after union |
| `MAX_POOL_SIZE` | 6000 | cap on pool size before cosine ranking (runtime safety valve) |
| `XGB_PARAMS` | — | XGBoost hyperparameters |
| `VALIDATION_FRACTION` | 0.20 | entity-level train/valid split |
| `PAIR_BATCH_SIZE` | 20,000 | candidate pairs featurized per batch |

## Known limitations / next things to try

- The XGBoost model currently ranks each candidate independently; it does
  not explicitly model "if S1 already has a very strong match, dampen
  weaker rivals." If a validation error analysis shows S1 entities with
  many near-tied candidates, a listwise re-ranking pass on top of the
  per-pair probabilities is worth adding.
- Address parsing is currently a single normalized string plus a
  regex-extracted ZIP/PIN; splitting street/city/state components (where
  present) into their own comparison features could add signal that the
  n-gram cosine feature partially but not fully captures.
