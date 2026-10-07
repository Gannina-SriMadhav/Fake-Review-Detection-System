"""Reference (training-time) distributions used for drift detection."""
from __future__ import annotations

import numpy as np

from src.features.style import FEATURE_NAMES, style_features

DRIFT_FEATURES = ["word_count", "char_count", "avg_word_len", "lexical_diversity", "exclamation_ratio",
                  "uppercase_ratio", "sentiment_compound", "punct_ratio"]


def feature_frame(texts):
    import pandas as pd
    rows = []
    for t in texts:
        f = style_features(t)
        rows.append({k: f[k] for k in DRIFT_FEATURES})
    return pd.DataFrame(rows)


def build_reference_stats(texts, proba_fake=None, max_n: int = 5000) -> dict:
    rng = np.random.default_rng(0)
    texts = list(texts)
    if len(texts) > max_n:
        texts = [texts[i] for i in rng.choice(len(texts), max_n, replace=False)]
    df = feature_frame(texts)
    out = {"features": {c: df[c].round(5).tolist() for c in df.columns}}
    if proba_fake is not None:
        p = np.asarray(proba_fake)
        out["fake_ratio"] = float((p >= 0.5).mean())
        out["confidence"] = np.where(p >= 0.5, p, 1 - p)[:max_n].round(4).tolist()
        out["proba_fake"] = p[:max_n].round(4).tolist()
    return out
