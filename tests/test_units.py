import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.loader import SchemaError, load_dataset, split_dataset, validate_schema
from src.evaluation.metrics import compute_metrics, confidence_level
from src.features.style import FEATURE_NAMES, StyleFeatures, style_features
from src.monitoring.drift import psi
from src.preprocessing.cleaning import basic_clean, normalize_text


def test_cleaning_keeps_style_and_strips_html():
    assert basic_clean("<b>GREAT</b>  product!!!\n") == "GREAT product!!!"
    assert "urltoken" in normalize_text("see http://x.com now")
    assert basic_clean(None) == "" and basic_clean(float("nan")) == ""
    assert normalize_text("Café ok") == "Café ok"


def test_style_features_columns_and_values():
    f = style_features("Great!!! Buy NOW 123 http://a.b 😀")
    assert set(f) == set(FEATURE_NAMES)
    assert f["url_count"] == 1 and f["emoji_count"] == 1 and f["exclamation_ratio"] > 0
    assert not any(np.isnan(v) for v in f.values())
    assert style_features("")["word_count"] == 0  # empty input must not crash


def test_style_transformer_shape():
    X = StyleFeatures().transform(["Good.", "A longer review with several words, honestly."])
    assert X.shape == (2, len(FEATURE_NAMES)) and np.isfinite(X).all()


def test_metrics_and_confidence_levels():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert confidence_level(0.9, .85, .7) == "HIGH" and confidence_level(0.75, .85, .7) == "MEDIUM" and confidence_level(0.55, .85, .7) == "LOW"


def test_psi_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=2000)
    assert psi(x, x) < 1e-6 and psi(x, x + 3) > 0.2


def test_schema_validation():
    with pytest.raises(SchemaError):
        validate_schema(pd.DataFrame({"a": [1]}), "text_", "label")


def test_data_quality_and_split_no_leakage():
    cfg = load_config()
    cfg["data"]["sample_size"] = 1500
    df, rep = load_dataset(cfg)
    assert {"text", "label", "group"} <= set(df.columns)
    assert df.label.isin([0, 1]).all() and df.text.str.len().min() > 0
    assert rep["fake_fraction"] == pytest.approx(0.5, abs=0.1)  # label-imbalance check
    assert not df.text.str.lower().duplicated().any()
    tr, va, te = split_dataset(df, cfg)
    assert not set(tr.group) & set(te.group) and not set(tr.text.str.lower()) & set(te.text.str.lower())
