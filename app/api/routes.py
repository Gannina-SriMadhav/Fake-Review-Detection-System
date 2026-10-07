from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.schemas.models import BatchRequest, BatchResponse, Prediction, ReviewRequest
from app.services.model_service import ModelNotReady
from src.monitoring.drift import drift_report

log = logging.getLogger("api")
router = APIRouter()


PRED_COUNT = Counter("fr_predictions_total", "Predictions served", ["label", "confidence_level"])
LATENCY = Histogram("fr_inference_seconds", "Inference latency", buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
LOW_CONF = Counter("fr_low_confidence_total", "Low-confidence predictions")
ERRORS = Counter("fr_errors_total", "Unhandled errors")


def svc(request: Request):
    s = request.app.state.service
    if not s.ready:
        raise HTTPException(503, "Model not available. Train a model first (python -m src.models.train).")
    return s


def _count(label, level, n=1):
    PRED_COUNT.labels(label, level).inc(n)
    if level == "LOW":
        LOW_CONF.inc(n)


@router.post("/predict", response_model=Prediction)
def predict(body: ReviewRequest, request: Request):
    s = svc(request)
    t = time.perf_counter()
    out = s.predict(body.review, explain=body.explain, model=body.model)
    LATENCY.observe(time.perf_counter() - t)
    _count(out["prediction"], out["confidence_level"])
    return out


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest, request: Request):
    s = svc(request)
    if len(body.reviews) > s.cfg["api"]["max_batch"]:
        raise HTTPException(413, f"Batch too large (max {s.cfg['api']['max_batch']}).")
    out = s.predict_batch(body.reviews, explain=body.explain, model=body.model)
    for r in out["results"]:
        _count(r["prediction"], r["confidence_level"])
    LATENCY.observe(out["processing_time_ms"] / 1000)
    return out


@router.post("/explain/lime")
def explain_lime(body: ReviewRequest, request: Request):
    return svc(request).lime(body.review, body.model)


@router.post("/predict/compare")
def predict_compare(body: ReviewRequest, request: Request):
    """Run the review through every available model and return each verdict side by side."""
    return {"results": svc(request).compare(body.review)}


GUIDE = {  # plain-language text only; every number shown with it comes from the evaluation report
    "logreg": ("Word scoreboard", "Gives every word and phrase a weight and adds them up. Simple, fast and easy to explain.", "Everyday use when you want a transparent answer."),
    "linear_svc": ("Sharpest boundary", "Finds the cleanest dividing line between genuine and fake writing patterns.", "Usually the most accurate of the simple models."),
    "naive_bayes": ("Quick counter", "Compares how often words appear in fake vs genuine reviews. Very fast, a bit less precise.", "Fast rough screening of lots of reviews."),
    "random_forest": ("Panel of trees", "Hundreds of small decision trees vote. Can notice non-obvious style patterns; slower and harder to explain.", "A second opinion that thinks differently."),
    "ensemble": ("Team vote", "Averages Logistic Regression, SVM and Naive Bayes so one model's quirk matters less.", "Borderline reviews where you want a steadier view."),
}


@router.get("/models")
def list_models(request: Request):
    s = svc(request)
    rep = s.report()
    out = []
    for k, n in s.meta["models"].items():
        g = GUIDE.get(k, (n, "", ""))
        c = rep["comparison"].get(k) if rep else None
        out.append({"key": k, "name": n, "nickname": g[0], "description": g[1], "best_for": g[2],
                    "is_default": k == s.meta["model_key"],
                    "test_f1_fake": c["test"]["f1_fake"] if c else None,
                    "ms_per_review": c["inference_ms_per_review"] if c else None})
    return out


@router.get("/health")
def health(request: Request):
    s = request.app.state.service
    return {"status": "ok" if s.ready else "degraded", "model_loaded": s.ready}


@router.get("/version")
def version(request: Request):
    s = request.app.state.service
    return {"api_version": "2.0.0", "model_version": s.meta["version"] if s.ready else None}


@router.get("/model-info")
def model_info(request: Request):
    s = svc(request)
    m = s.meta
    rep = s.report()
    test = rep["comparison"][m["model_key"]]["test"] if rep else None
    return {"model": m["model_name"], "version": m["version"], "trained_at": m["trained_at"],
            "selection": f"best {m['selection_metric']} on {m['selection_split']} split",
            "calibration": m["calibration"], "confidence_thresholds": m["confidence_thresholds"],
            "n_train": m["n_train"], "n_test": m["n_test"], "test_metrics": test}


@router.get("/performance")
def performance(request: Request):
    rep = svc(request).report()
    if not rep:
        raise HTTPException(404, "Not evaluated yet.")
    return rep


@router.get("/analytics")
def analytics(request: Request):
    df = svc(request).store.frame()
    if df.empty:
        return {"total": 0}
    import numpy as np
    return {"total": int(len(df)), "genuine": int((df.label == "GENUINE").sum()), "fake": int((df.label == "FAKE").sum()),
            "avg_confidence": float(df.confidence.mean()), "low_confidence": int((df.confidence_level == "LOW").sum()),
            "avg_latency_ms": float(df.latency_ms.mean()),
            "confidence": df.confidence.round(3).tolist(), "sentiment": df.sentiment.round(3).tolist(),
            "word_count": df.word_count.tolist(), "label": df.label.tolist(), "ts": df.ts.tolist(),
            "p_fake": df.p_fake.round(3).tolist()}


@router.get("/monitoring/drift")
def drift(request: Request):
    import json
    s = svc(request)
    p = s.models_dir / "reference_stats.json"
    if not p.exists():
        raise HTTPException(404, "Reference statistics missing.")
    return drift_report(json.loads(p.read_text()), s.store.frame(), s.cfg)


@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
