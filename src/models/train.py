"""Training pipeline: load -> clean/dedupe -> leakage-safe split -> fit models -> calibrate -> compare -> reports.

Every trained model is saved (models/<key>.joblib) so the app can use any of them or compare them.
The *default* model is the best F1(fake) on the VALIDATION split; test is scored once for reporting.

Run:  python -m src.models.train [--sample N] [--cv] [--no-mlflow]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import load_config, resolve
from src.data.loader import load_dataset, split_dataset
from src.evaluation.error_analysis import error_analysis
from src.evaluation.metrics import compute_metrics, confidence_histogram, curves
from src.evaluation.model_card import write_model_card
from src.explainability.similarity import build_index
from src.features.style import FEATURE_NAMES
from src.models import pipelines as P
from src.monitoring.reference import build_reference_stats

log = logging.getLogger("train")

# Fixed settings (picked earlier on the validation split; no per-run grid search to keep training quick)
PARAMS = {"logreg": {"C": 16}, "linear_svc": {"C": 1.0}, "naive_bayes": {"alpha": 0.1}, "random_forest": {}}


def _ms_per_review(model, texts, n=200):
    texts = list(texts[:n])
    model.predict_proba(texts[:5])
    t = time.perf_counter()
    model.predict_proba(texts)
    return (time.perf_counter() - t) / len(texts) * 1000


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--cv", action="store_true", help="also run grouped cross-validation (slow)")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    if args.sample:
        cfg["data"]["sample_size"] = args.sample
    seed, tr = cfg["project"]["seed"], cfg["training"]
    models_dir, reports_dir = resolve(cfg["paths"]["models_dir"]), resolve(cfg["paths"]["reports_dir"])
    models_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)

    df, data_report = load_dataset(cfg)
    train, val, test = split_dataset(df, cfg)
    data_report["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    (reports_dir / "data_report.json").write_text(json.dumps(data_report, indent=2))
    log.info("data: %s", data_report)
    Tr, Va, Te = (x["text"].tolist() for x in (train, val, test))

    # features are fitted once per kind on TRAIN only, then shared by the models that use them
    feats, mats = {}, {}
    results, fitted, val_p, test_p = {}, {}, {}, {}
    for name in tr["candidates"]:
        t0 = time.time()
        kind = P.FEATURE_KIND[name]
        if kind not in feats:
            feats[kind] = P.feature_pipeline(cfg["features"], kind).fit(Tr, train["label"])
            mats[kind] = feats[kind].transform(Tr)
        clf = P.calibrated(P.classifier(name, seed, PARAMS[name]), tr["calibration"], tr["calibration_cv"])
        clf.fit(mats[kind], train["label"])
        pipe = Pipeline([("features", feats[kind]), ("clf", clf)])
        pv, pt = pipe.predict_proba(Va)[:, 1], pipe.predict_proba(Te)[:, 1]
        res = {"display_name": P.DISPLAY[name], "params": PARAMS[name], "val": compute_metrics(val["label"], pv),
               "test": compute_metrics(test["label"], pt), "inference_ms_per_review": _ms_per_review(pipe, Te),
               "train_seconds": time.time() - t0}
        if args.cv:
            cvp = Pipeline([("features", P.feature_pipeline(cfg["features"], kind)), ("clf", P.classifier(name, seed, PARAMS[name]))])
            sc = cross_validate(cvp, Tr, train["label"], scoring="f1", n_jobs=tr["cv_folds"],
                                cv=StratifiedGroupKFold(tr["cv_folds"], shuffle=True, random_state=seed), groups=train["group"])
            res["cv_f1_fake_mean"], res["cv_f1_fake_std"] = float(sc["test_score"].mean()), float(sc["test_score"].std())
        results[name], fitted[name], val_p[name], test_p[name] = res, pipe, pv, pt
        log.info("%-14s val F1=%.4f  test F1=%.4f  (%.0fs)", name, res["val"]["f1_fake"], res["test"]["f1_fake"], time.time() - t0)

    members = [n for n in ("logreg", "linear_svc", "naive_bayes") if n in fitted]
    if tr.get("ensemble") and len(members) >= 2:
        ens = P.SoftVotingEnsemble([fitted[n] for n in members])
        pv, pt = ens.predict_proba(Va)[:, 1], ens.predict_proba(Te)[:, 1]
        results["ensemble"] = {"display_name": P.DISPLAY["ensemble"], "params": {"members": members},
                               "val": compute_metrics(val["label"], pv), "test": compute_metrics(test["label"], pt),
                               "inference_ms_per_review": _ms_per_review(ens, Te)}
        fitted["ensemble"], val_p["ensemble"], test_p["ensemble"] = ens, pv, pt
        log.info("%-14s val F1=%.4f  test F1=%.4f", "ensemble", results["ensemble"]["val"]["f1_fake"], results["ensemble"]["test"]["f1_fake"])

    metric = tr["selection_metric"]
    prod = max(fitted, key=lambda k: results[k]["val"][metric])
    for k in results:
        results[k]["is_production"] = k == prod
    log.info("Default model: %s (val %s=%.4f)", prod, metric, results[prod]["val"][metric])

    for k, m in fitted.items():
        joblib.dump(m, models_dir / f"{k}.joblib", compress=3)
    build_index(pd.concat([train, val])["text"].tolist(), models_dir / "similarity_index.joblib")
    (models_dir / "reference_stats.json").write_text(json.dumps(build_reference_stats(Tr, val_p[prod])))
    meta = {"model_key": prod, "model_name": P.DISPLAY[prod], "version": dt.datetime.utcnow().strftime("%Y.%m.%d.%H%M"),
            "trained_at": dt.datetime.utcnow().isoformat() + "Z", "selection_metric": metric, "selection_split": "validation",
            "confidence_thresholds": cfg["confidence"], "calibration": tr["calibration"], "n_train": len(train), "n_val": len(val),
            "n_test": len(test), "dataset": cfg["data"]["path"], "style_features": FEATURE_NAMES, "params": results[prod]["params"],
            "models": {k: P.DISPLAY[k] for k in fitted}}
    (models_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))

    ea = error_analysis(test, test_p[prod], prod)
    report = {"meta": meta, "comparison": results, "curves": {k: curves(test["label"].values, v) for k, v in test_p.items()},
              "confidence_hist": {k: confidence_histogram(v) for k, v in test_p.items()}, "data": data_report, "error_analysis": ea}
    (reports_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    write_model_card(meta, results[prod], data_report, ea, reports_dir / "model_card.md")
    pd.DataFrame([{"model": v["display_name"], "key": k, **{m: v["test"][m] for m in ("accuracy", "precision_fake", "recall_fake", "f1_fake", "roc_auc", "pr_auc", "brier")},
                   "inference_ms": v["inference_ms_per_review"], "default": v["is_production"]} for k, v in results.items()]
                 ).to_csv(reports_dir / "model_comparison.csv", index=False)

    if not args.no_mlflow:
        try:
            log_mlflow(cfg, results, prod, models_dir, reports_dir)
        except Exception as e:  # tracking must never break training
            log.warning("MLflow logging failed: %s", e)
    log.info("Done.")


def log_mlflow(cfg, results, prod, models_dir, reports_dir):
    import os
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["paths"]["mlflow_uri"]))
    mlflow.set_experiment(cfg["paths"]["mlflow_experiment"])
    for k, v in results.items():
        with mlflow.start_run(run_name=k):
            mlflow.log_params({"model": k, "calibration": cfg["training"]["calibration"], "seed": cfg["project"]["seed"],
                               **{f"p_{a}": b for a, b in v["params"].items() if not isinstance(b, list)}})
            for split in ("val", "test"):
                for mk, mv in v[split].items():
                    if isinstance(mv, float):
                        mlflow.log_metric(f"{split}_{mk}", mv)
            mlflow.log_metric("inference_ms", v["inference_ms_per_review"])
            mlflow.log_artifact(str(models_dir / f"{k}.joblib"))
            if k == prod:
                mlflow.set_tag("default_model", "true")
                mlflow.log_artifacts(str(reports_dir), "reports")


if __name__ == "__main__":
    main()
