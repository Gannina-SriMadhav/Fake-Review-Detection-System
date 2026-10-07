"""False-positive / false-negative analysis, computed from real test predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.style import style_features


def _profile(texts):
    f = pd.DataFrame([style_features(t) for t in texts])
    return {c: round(float(f[c].mean()), 4) for c in
            ("word_count", "exclamation_ratio", "uppercase_ratio", "lexical_diversity", "sentiment_compound")} if len(f) else {}


def error_analysis(test: pd.DataFrame, proba_fake, model_key: str, n_examples: int = 8) -> dict:
    p = np.asarray(proba_fake)
    pred = (p >= 0.5).astype(int)
    y = test["label"].values
    fp = test[(pred == 1) & (y == 0)].assign(p_fake=p[(pred == 1) & (y == 0)])
    fn = test[(pred == 0) & (y == 1)].assign(p_fake=p[(pred == 0) & (y == 1)])
    tp_t, tn_t = test.text[(pred == 1) & (y == 1)], test.text[(pred == 0) & (y == 0)]
    words = test.text.str.split().str.len()
    err = pred != y
    bins = pd.cut(words, [0, 5, 15, 40, 100, 10_000], labels=["<=5", "6-15", "16-40", "41-100", ">100"])
    by_len = {str(b): {"n": int((bins == b).sum()), "error_rate": round(float(err[(bins == b).values].mean()), 4)
                       if (bins == b).any() else None} for b in bins.cat.categories}
    conf = np.where(pred == 1, p, 1 - p)
    return {
        "model": model_key, "n_false_positive": int(len(fp)), "n_false_negative": int(len(fn)),
        "false_positives": [{"text": r.text[:400], "p_fake": round(float(r.p_fake), 3)}
                            for r in fp.sort_values("p_fake", ascending=False).head(n_examples).itertuples()],
        "false_negatives": [{"text": r.text[:400], "p_fake": round(float(r.p_fake), 3)}
                            for r in fn.sort_values("p_fake").head(n_examples).itertuples()],
        "error_rate_by_length": by_len,
        "mean_confidence_correct": round(float(conf[~err].mean()), 4),
        "mean_confidence_errors": round(float(conf[err].mean()), 4) if err.any() else None,
        "style_profile": {"false_positives": _profile(fp.text.tolist()), "false_negatives": _profile(fn.text.tolist()),
                          "true_positives": _profile(tp_t.tolist()), "true_negatives": _profile(tn_t.tolist())},
        "note": "Profiles are descriptive averages of test reviews in each outcome group; they illustrate where errors "
                "concentrate and are not causal explanations.",
    }
