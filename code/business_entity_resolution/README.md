# Business Entity Resolution Pipeline

End-to-end ML pipeline for the ML Challenge 2026: Business Entity Resolution.

## Overview

Given business records from 3 independent data sources with noisy and inconsistent fields, this pipeline determines which records across sources refer to the same real-world business entity.

**Pipeline stages:**
1. **Data Loading** — Read and normalise all source TSVs
2. **Blocking** — Country-aware TF-IDF blocking to reduce the O(N²) comparison space
3. **Feature Engineering** — String similarity features (Jaro-Winkler, Levenshtein, Jaccard, containment, sorted-token JW) on names and addresses
4. **Model Training** — XGBoost classifier trained on the provided ground truth
5. **Inference** — Predict matches on test pairs with an auto-tuned threshold optimised for F₀.₅

## Requirements

- Python 3.8+
- ~16 GB RAM recommended (dataset is ~2M S1 × ~10M S2+S3 entities)

## Setup

```bash
pip install -r requirements.txt
```

## Running the Pipeline

From the `src/` directory:

```bash
python pipeline.py \
    --train_dir ../../../dataset/train \
    --test_dir  ../../../dataset/test \
    --output_dir ../../../output
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--train_dir` | `../../../dataset/train` | Path to training data |
| `--test_dir` | `../../../dataset/test` | Path to test data |
| `--output_dir` | `../../../output` | Path for output TSV files |
| `--top_k` | `10` | Number of blocking candidates per S1 entity |
| `--threshold` | *auto-tuned* | Override the match probability threshold |
| `--batch_size` | `50000` | Batch size for TF-IDF similarity computation |
| `--skip_validation` | `false` | Skip threshold tuning (use default 0.6) |

## Outputs

Both files are placed in `output/`:

- **`matching_results.tsv`** — Final predicted matches (scored on the leaderboard)
- **`candidate_pairs.tsv`** — Blocking candidate set fed to the model

## Validation

```bash
python ../../../utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir ../../../dataset/test
```
