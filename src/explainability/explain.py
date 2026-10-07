"""Model-faithful explanations.

Occlusion: each word is removed in turn and the *deployed model* is re-queried; the change in
calibrated P(FAKE) is that word's contribution. This is technically supported for any model
(linear, tree, ensemble) - we never invent signals. Statements are phrased as "contributed to
the model's prediction", never as proof.
"""
from __future__ import annotations

import re

import numpy as np

MAX_TOKENS = 60


def occlusion(model, text: str, top_k: int = 6) -> dict:
    toks = text.split()
    if not toks:
        return {"method": "occlusion", "toward_fake": [], "toward_genuine": []}
    cap = toks[:MAX_TOKENS]
    variants = [text] + [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(cap))]
    probs = model.predict_proba(variants)[:, 1]
    base, deltas = probs[0], base_delta(probs)
    items = [{"token": re.sub(r"^\W+|\W+$", "", cap[i]) or cap[i], "effect": float(deltas[i])} for i in range(len(cap))]
    agg: dict[str, float] = {}
    for it in items:
        agg[it["token"].lower()] = agg.get(it["token"].lower(), 0.0) + it["effect"]
    ranked = sorted(agg.items(), key=lambda kv: kv[1])
    fake = [{"token": k, "effect": round(v, 4)} for k, v in reversed(ranked) if v > 0.002][:top_k]
    gen = [{"token": k, "effect": round(v, 4)} for k, v in ranked if v < -0.002][:top_k]
    return {"method": "occlusion", "base_p_fake": float(base), "truncated": len(toks) > MAX_TOKENS,
            "toward_fake": fake, "toward_genuine": gen}


def base_delta(probs):
    # effect of word i on P(FAKE) = P(full) - P(without word i)
    return probs[0] - probs[1:]


def lime_explain(model, text: str, num_features: int = 8, num_samples: int = 400) -> dict:
    from lime.lime_text import LimeTextExplainer
    ex = LimeTextExplainer(class_names=["GENUINE", "FAKE"], random_state=0)
    e = ex.explain_instance(text, model.predict_proba, num_features=num_features, num_samples=num_samples)
    w = e.as_list(label=1)
    return {"method": "lime", "toward_fake": [{"token": k, "effect": round(v, 4)} for k, v in w if v > 0],
            "toward_genuine": [{"token": k, "effect": round(v, 4)} for k, v in w if v < 0]}


def describe(label: str, ex: dict, stats: dict, sim: dict) -> list[str]:
    """Human-readable signal lines, each backed by something the model/analysis actually produced."""
    out = []
    side = ex["toward_fake"] if label == "FAKE" else ex["toward_genuine"]
    if side:
        words = ", ".join(f"'{s['token']}'" for s in side[:4])
        out.append(f"Wording that pushed the model toward {label}: {words}.")
    other = ex["toward_genuine"] if label == "FAKE" else ex["toward_fake"]
    if other:
        words = ", ".join(f"'{s['token']}'" for s in other[:3])
        out.append(f"Wording that pushed against it (toward {'GENUINE' if label == 'FAKE' else 'FAKE'}): {words}.")
    if not out:
        out.append("No single word had a strong individual effect; the prediction reflects the overall wording/style pattern.")
    if sim.get("warning"):
        out.append(f"Similarity layer: {sim['level'].replace('_', ' ').lower()} match found "
                   f"(cosine {sim['max_similarity']:.2f}). This is a supporting signal and does not change the label.")
    if stats["word_count"] < 6:
        out.append("The review is very short, so there is little text for the model to base its judgement on.")
    return out
