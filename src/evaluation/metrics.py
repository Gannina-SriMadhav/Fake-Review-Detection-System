"""Evaluation helpers. Positive class = FAKE (1)."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """ECE on the probability of the *predicted* class (what the UI shows as confidence)."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y, proba_fake, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    p = np.asarray(proba_fake, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_fake": float(precision_score(y, pred, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, zero_division=0)),
        "precision_genuine": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_genuine": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "f1_genuine": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # FP = genuine flagged FAKE ; FN = fake passed as GENUINE
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
    }


def curves(y, proba_fake, max_points: int = 200) -> dict:
    y = np.asarray(y)
    p = np.asarray(proba_fake)

    def thin(*arrs):
        n = len(arrs[0])
        if n <= max_points:
            return [a.tolist() for a in arrs]
        idx = np.linspace(0, n - 1, max_points).astype(int)
        return [a[idx].tolist() for a in arrs]

    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    fpr, tpr = thin(fpr, tpr)
    pr, rc = thin(pr, rc)
    return {"roc": {"fpr": fpr, "tpr": tpr}, "pr": {"precision": pr, "recall": rc},
            "calibration": {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()}}


def confidence_level(conf: float, high: float, medium: float) -> str:
    return "HIGH" if conf >= high else "MEDIUM" if conf >= medium else "LOW"


def confidence_histogram(y_proba_fake, bins: int = 10) -> dict:
    p = np.asarray(y_proba_fake)
    conf = np.where(p >= 0.5, p, 1 - p)
    counts, edges = np.histogram(conf, bins=bins, range=(0.5, 1.0))
    return {"edges": edges.tolist(), "counts": counts.tolist()}
