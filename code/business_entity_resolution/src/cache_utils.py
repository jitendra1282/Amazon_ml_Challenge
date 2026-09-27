"""
Disk caching so a re-run (or Stage 2 running after Stage 1 in a separate
process) never re-fits the vectorizer/IDF or re-normalizes data it
already computed. This was explicitly called out as a time sink in the
prior notebooks, where Stage 2 reloaded and re-normalized everything
independently of Stage 1.
"""

from __future__ import annotations

import os
import pickle

import config


def save_vectorizer(vectorizer, idf, path: str = config.VECTORIZER_CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "idf": idf}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_vectorizer(path: str = config.VECTORIZER_CACHE_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["vectorizer"], obj["idf"]


def save_normalized(df, name: str, cache_dir: str = config.NORMALIZED_CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}.pkl")
    df.to_pickle(path)
    return path


def load_normalized(name: str, cache_dir: str = config.NORMALIZED_CACHE_DIR):
    path = os.path.join(cache_dir, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    return __import__("pandas").read_pickle(path)
