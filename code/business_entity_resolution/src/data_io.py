"""
Load + normalize the raw TSVs, with a disk cache so re-running the
pipeline (or running Stage 2 on its own) never re-does normalization for
data that hasn't changed.
"""

from __future__ import annotations

import gc

import pandas as pd

import cache_utils
import config
from normalization import normalize_dataframe


def load_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_and_normalize(path: str, cache_name: str, use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        cached = cache_utils.load_normalized(cache_name)
        if cached is not None:
            print(f"  [cache hit] {cache_name} ({len(cached):,} rows)")
            return cached

    df = load_tsv(path)
    normalize_dataframe(df)
    df = df[config.KEEP_COLS].copy()
    gc.collect()

    if use_cache:
        cache_utils.save_normalized(df, cache_name)

    print(f"  [normalized] {cache_name} ({len(df):,} rows)")
    return df


def load_all_train(use_cache: bool = True):
    s1 = load_and_normalize(config.TRAIN_S1_PATH, "train_s1", use_cache)
    s2 = load_and_normalize(config.TRAIN_S2_PATH, "train_s2", use_cache)
    s3 = load_and_normalize(config.TRAIN_S3_PATH, "train_s3", use_cache)
    gt = load_tsv(config.GROUND_TRUTH_PATH)
    return s1, s2, s3, gt


def load_all_test(use_cache: bool = True):
    s1 = load_and_normalize(config.TEST_S1_PATH, "test_s1", use_cache)
    s2 = load_and_normalize(config.TEST_S2_PATH, "test_s2", use_cache)
    s3 = load_and_normalize(config.TEST_S3_PATH, "test_s3", use_cache)
    return s1, s2, s3


def concat_s23(s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    return s23
