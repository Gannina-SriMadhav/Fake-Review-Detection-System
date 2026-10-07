"""Model Robustness Testing.

Runs difficult / simple / edge-case reviews through the production model and records what it does.
NO expected labels are enforced - this is an inspection tool to find weaknesses, not a pass/fail oracle.

Run: python -m src.evaluation.robustness
"""
from __future__ import annotations

import json

import pandas as pd

CASES = {
    "simple_short": ["Good.", "Nice.", "Good product.", "Works well.", "Very useful.", "I liked it.", "Excellent quality.",
                     "Not bad.", "Average product.", "Okay.", "Average.", "Works as expected.", "This is good.",
                     "Nice product, works well.", "Very useful product."],
    "positive_enthusiastic": ["Best purchase ever.", "Absolutely amazing!!!", "Absolutely amazing!!! Best product ever!!! Buy this now!!!",
                              "Very good product, satisfied."],
    "negative": ["Terrible product.", "Not worth the money.", "Broke after two days, very disappointed."],
    "neutral_specific": ["I bought this last week and it works fine.", "This product is good but the packaging was damaged.",
                         "Arrived on time. Does the job, nothing special."],
    "long_detailed": [
        "I have been using this vacuum for about two months on hardwood floors and a small rug. Suction is solid and the battery "
        "lasts roughly 35 minutes on the low setting. The dustbin is a bit small so I empty it twice per cleaning, but it is easy to "
        "detach. The charging dock feels flimsy though it has not caused any problems so far."],
    "promotional_looking": ["Great product! Great quality! Great price! Highly recommend to everyone, five stars, buy it today!"],
    "poorly_written": ["gud prodct work fine thx", "it is not working proper but ok price"],
    "repeated_text": ["Good product. Good product. Good product. Good product."],
    "edge_inputs": ["a", "!!!", "12345", "👍👍👍", "<b>nice</b> http://example.com", "x " * 400],
}


def run(model=None) -> list[dict]:
    if model is None:
        import json
        import joblib
        from src.config import load_config, resolve
        d = resolve(load_config()["paths"]["models_dir"])
        model = joblib.load(d / f"{json.loads((d / 'model_meta.json').read_text())['model_key']}.joblib")
    rows = []
    for cat, texts in CASES.items():
        p = model.predict_proba(texts)[:, 1]
        for t, pf in zip(texts, p):
            lab = "FAKE" if pf >= .5 else "GENUINE"
            rows.append({"category": cat, "review": t[:90], "prediction": lab, "confidence": round(float(pf if lab == "FAKE" else 1 - pf), 3),
                         "p_fake": round(float(pf), 3)})
    return rows


def main():
    from src.config import load_config, resolve
    rows = run()
    df = pd.DataFrame(rows)
    summary = df.groupby("category").agg(n=("prediction", "size"), pct_fake=("prediction", lambda s: round((s == "FAKE").mean() * 100, 1)),
                                         mean_conf=("confidence", "mean")).round(3)
    out = resolve(load_config()["paths"]["reports_dir"])
    (out / "robustness.json").write_text(json.dumps({"cases": rows, "summary": summary.reset_index().to_dict("records")}, indent=2))
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(df.to_string(index=False)); print(); print(summary)


if __name__ == "__main__":
    main()
