# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We built a country-aware TF-IDF blocking + XGBoost classification pipeline for business entity resolution across 3 noisy data sources. The pipeline normalises business names (expanding abbreviations like Corp→Corporation, Pvt→Private) and addresses (St→Street, Rd→Road), uses character n-gram TF-IDF vectors to identify top-K candidates per S1 entity within each country, computes rich string-similarity features (Jaro-Winkler, sorted-token JW, Levenshtein, Jaccard, containment), and trains an XGBoost classifier with class imbalance handling to predict matches. The threshold is auto-tuned to maximise the precision-heavy F₀.₅ metric.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** ~2.2M S1 entities, ~5M S2, ~5.3M S3 in training; ~1.7M S1, ~4.9M S2, ~5.1M S3 in test.
- **Noise patterns:** Name abbreviations (Corp, Inc, Ltd, Pvt, LLC), punctuation (&/and), DBA names, transliterations (Hindi Devanagari script in Indian records), word-order transpositions.
- **Address noise:** Abbreviations (St, Rd, Ave), landmark-based references ("Near SBI ATM"), missing PIN codes, reordered components, transliteration variants.
- **Open-set countries:** Training has US + India; test adds France. Pipeline must not hard-code country labels.

### 2.2 Solution Strategy

**Approach Type:** Country-Aware Blocking + XGBoost Classifier  
**Core Innovation:** Country-partitioned TF-IDF blocking to reduce the comparison space while preserving recall; rich symmetric + asymmetric string similarity features; precision-optimised threshold tuning for F₀.₅.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Character n-gram (2-4) TF-IDF vectors on normalised (name + address), with `sublinear_tf=True` and `max_features=200,000`.
- **Strategy:** Country-aware blocking — for each country label, block S1 vs S2+S3 within that country. A global fallback catches S1 entities with missing/unknown country.
- **Top-K:** 10 candidates per S1 entity (configurable via `--top_k`).
- **Computation:** Batched cosine similarity (batch_size=50,000 S1 entities) with `argpartition` for efficient top-K extraction.
- **How we ensured true matches were not lost:** (a) Character n-grams are robust to typos and abbreviation differences; (b) country-aware partitioning ensures same-country entities are always compared; (c) global fallback for entities without country match.

---

## 4. Matching Model

**Features used:**
- **Name features:** Jaro-Winkler similarity, sorted-token Jaro-Winkler (order-invariant), Levenshtein distance, token Jaccard similarity, containment similarity, length difference, length ratio.
- **Address features:** Jaro-Winkler similarity, sorted-token Jaro-Winkler, Levenshtein distance, token Jaccard similarity, containment similarity, length difference.
- **Other:** Exact country match (binary), TF-IDF cosine distance from blocking.

**Model type:** XGBoost (`XGBClassifier`) with `scale_pos_weight` for class imbalance, `tree_method='hist'` for speed, 300 estimators, max_depth=7.

**Threshold selection method:** Grid search over thresholds [0.30, 0.85] in steps of 0.05, evaluated on macro-averaged F₀.₅ on the training set. The threshold that maximises F₀.₅ is selected. Singletons (S1 entities with no true matches) are included in the F₀.₅ average.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** Businesses with similar names at different addresses; franchises/chains with identical names in different cities.
- **Common false negatives (missed matches):** Heavily abbreviated or transliterated names that differ significantly in character representation; addresses with landmark-only references.

---

## 6. Conclusion
Our pipeline balances scalability with matching quality through country-aware blocking and precision-tuned classification. Key achievements: handles multi-million entity datasets with batched computation, gracefully handles the unseen France country in test data, and optimises for the precision-heavy F₀.₅ metric. Lessons learned: character n-grams are more robust than word-level TF-IDF for noisy entity names; sorted-token similarity features are critical for handling word-order transpositions.

---

## Appendix

### A. Code Artefacts
All source code is in `code/business_entity_resolution/src/pipeline.py`. The entry point is:

```bash
cd code/business_entity_resolution/src
python pipeline.py --train_dir ../../../dataset/train --test_dir ../../../dataset/test --output_dir ../../../output
```

This produces both `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Dependencies are listed in `code/business_entity_resolution/requirements.txt`.

### B. Additional Results
[Include any additional charts, graphs, or detailed results.]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
