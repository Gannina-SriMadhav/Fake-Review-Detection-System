"""Drift detection: PSI + KS per feature, prediction/confidence drift. Flags only - never auto-retrains."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.monitoring.reference import DRIFT_FEATURES, feature_frame


def psi(ref, cur, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref_stats: dict, log_df: pd.DataFrame, cfg: dict) -> dict:
    m = cfg["monitoring"]
    n = len(log_df)
    out = {"n_current": n, "min_samples": m["min_samples"], "features": [], "drift_detected": False,
           "status": "INSUFFICIENT_DATA" if n < m["min_samples"] else "OK", "message": ""}
    if n < m["min_samples"]:
        out["message"] = f"Need at least {m['min_samples']} logged predictions to assess drift (have {n})."
        return out
    cur = feature_frame(log_df["text"].tolist())
    for f in DRIFT_FEATURES:
        r = ref_stats["features"][f]
        p_ = psi(r, cur[f])
        ks = ks_2samp(r, cur[f])
        flag = p_ >= m["psi_threshold"] and ks.pvalue < m["ks_pvalue_threshold"]
        out["features"].append({"feature": f, "psi": round(p_, 4), "ks_stat": round(float(ks.statistic), 4),
                                "ks_pvalue": float(ks.pvalue), "drift": bool(flag)})
    pred = {}
    if "proba_fake" in ref_stats:
        cur_fake = float((log_df["label"] == "FAKE").mean())
        pred["fake_ratio_reference"], pred["fake_ratio_current"] = ref_stats["fake_ratio"], cur_fake
        ks = ks_2samp(ref_stats["confidence"], log_df["confidence"])
        pred["confidence_ks_stat"], pred["confidence_ks_pvalue"] = round(float(ks.statistic), 4), float(ks.pvalue)
        pred["prediction_drift"] = bool(abs(cur_fake - ref_stats["fake_ratio"]) > 0.2)
    pred["low_confidence_rate"] = float((log_df["confidence_level"] == "LOW").mean())
    out["prediction"] = pred
    feat_drift = [f["feature"] for f in out["features"] if f["drift"]]
    out["drift_detected"] = bool(feat_drift or pred.get("prediction_drift"))
    out["status"] = "DRIFT" if out["drift_detected"] else "OK"
    out["message"] = ("Potential data drift detected" + (f" in: {', '.join(feat_drift)}" if feat_drift else "")
                      + ". Review recent inputs; retraining is a manual, controlled step (see retrain workflow)."
                      ) if out["drift_detected"] else "No drift above configured thresholds."
    if pred["low_confidence_rate"] > m["low_confidence_rate_alert"]:
        out["message"] += f" High low-confidence rate ({pred['low_confidence_rate']:.0%})."
    return out


def evidently_html(ref_stats: dict, log_df: pd.DataFrame, path: str) -> bool:
    """Optional Evidently data-drift HTML report; returns False if Evidently is unavailable."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        ref = pd.DataFrame(ref_stats["features"])
        cur = feature_frame(log_df["text"].tolist())
        Report([DataDriftPreset()]).run(cur, ref).save_html(path)
        return True
    except Exception:
        return False
