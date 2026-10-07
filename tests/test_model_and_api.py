import io
import json
import time

import numpy as np
import pytest

from tests.conftest import needs_model, ROOT

pytestmark = needs_model

SIMPLE = ["Good.", "Nice.", "Good product.", "Works well.", "Very useful.", "I liked it.", "Excellent quality.",
          "Not bad.", "Average product.", "Okay."]


# ------------------------------------------------------------ ML tests
def test_artifacts_exist():
    for f in ("model_meta.json", "similarity_index.joblib", "reference_stats.json"):
        assert (ROOT / "models" / f).exists(), f
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    assert meta["selection_split"] == "validation" and len(meta["models"]) >= 2
    for k in meta["models"]:
        assert (ROOT / "models" / f"{k}.joblib").exists(), k


def test_predictions_valid(model):
    p = model.predict_proba(SIMPLE + ["a", "!!!", "x " * 1000, "👍", "<b>hi</b>"])
    assert p.shape[1] == 2 and np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6) and (p >= 0).all() and (p <= 1).all()


def test_latency(model):
    model.predict_proba(["warm up"])
    t = time.perf_counter(); model.predict_proba(["This is a reasonable length review of a product."] * 20)
    assert (time.perf_counter() - t) / 20 < 0.25  # seconds per review, generous CI bound


def test_no_rule_based_override(model):
    """Output must be the model's own probability, not a length/keyword rule: it varies continuously."""
    p = model.predict_proba(SIMPLE)[:, 1]
    assert len(set(np.round(p, 4))) > 3


def test_reported_metrics_not_placeholder():
    rep = json.loads((ROOT / "reports" / "evaluation.json").read_text())
    for v in rep["comparison"].values():
        assert 0 <= v["test"]["f1_fake"] <= 1 and v["test"]["accuracy"] < 1.0  # never claim a perfect score
    assert sum(v["is_production"] for v in rep["comparison"].values()) == 1


# ------------------------------------------------------------ API tests
def test_health_version_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert "model_version" in client.get("/version").json()
    assert client.get("/model-info").json()["model"]
    assert b"fr_predictions_total" in client.get("/metrics").content or client.get("/metrics").status_code == 200


def test_predict_schema(client):
    r = client.post("/predict", json={"review": "I bought this last week and it works fine."})
    assert r.status_code == 200
    d = r.json()
    assert d["prediction"] in ("GENUINE", "FAKE") and 0.5 <= d["confidence"] <= 1
    assert d["confidence_level"] in ("HIGH", "MEDIUM", "LOW")
    assert abs(sum(d["probabilities"].values()) - 1) < 1e-3
    assert d["processing_time_ms"] >= 0 and isinstance(d["explanation"], list) and "similarity_warning" in d
    assert "not proof" in d["disclaimer"]
    assert "Traceback" not in r.text


def test_low_confidence_has_warning(client, model):
    for t in SIMPLE + ["Okay, I guess.", "It is a thing."]:
        d = client.post("/predict", json={"review": t}).json()
        assert (d["confidence_level"] == "LOW") == (d["warning"] is not None)


def test_validation_and_errors(client):
    assert client.post("/predict", json={"review": "   "}).status_code == 422
    assert client.post("/predict", json={"review": "x" * 6000}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", content=b"not json", headers={"content-type": "application/json"}).status_code == 422
    r = client.post("/predict/batch", json={"reviews": ["ok", ""]})
    assert r.status_code == 422 and "Traceback" not in r.text
    assert client.post("/predict/batch", json={"reviews": []}).status_code == 422


def test_unusual_inputs_do_not_crash(client):
    for t in ["👍👍👍", "<script>alert(1)</script>", "\x00\x01 weird ‮ text", "ñandú ☃ 日本語のレビュー", "." * 4000]:
        assert client.post("/predict", json={"review": t}).status_code == 200


def test_batch(client):
    r = client.post("/predict/batch", json={"reviews": SIMPLE})
    d = r.json()
    assert r.status_code == 200 and d["count"] == len(SIMPLE)
    assert all(x["prediction"] in ("GENUINE", "FAKE") for x in d["results"])


def test_model_selection_and_compare(client):
    ms = client.get("/models").json()
    assert len(ms) >= 2 and sum(m["is_default"] for m in ms) == 1
    for m in ms:
        d = client.post("/predict", json={"review": "Works fine for the price, arrived on time.", "model": m["key"], "explain": False}).json()
        assert d["model_key"] == m["key"] and d["prediction"] in ("GENUINE", "FAKE")
    c = client.post("/predict/compare", json={"review": "Works fine for the price, arrived on time."}).json()["results"]
    assert len(c) == len(ms)
    assert client.post("/predict", json={"review": "ok fine", "model": "nope"}).status_code == 422


def test_web_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Fake Review Intelligence" in r.text and "streamlit" not in r.text.lower()
