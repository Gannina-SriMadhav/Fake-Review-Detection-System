"""Single place where the production model is loaded and used."""
from __future__ import annotations

import json
import threading
import time

import joblib
import numpy as np

from src.config import load_config, resolve
from src.evaluation.metrics import confidence_level
from src.explainability.explain import describe, lime_explain, occlusion
from src.explainability.similarity import SimilarityIndex
from src.features.style import style_features
from src.monitoring.store import PredictionStore

DISCLAIMER = "Model prediction only - an automated estimate, not proof that a review is fraudulent."


class ModelNotReady(RuntimeError):
    pass


class ModelService:
    def __init__(self, config_path=None):
        self.cfg = load_config(config_path)
        self.models_dir = resolve(self.cfg["paths"]["models_dir"])
        self.reports_dir = resolve(self.cfg["paths"]["reports_dir"])
        self.model = self.meta = self.sim = None
        self.models: dict = {}
        self.store = PredictionStore(resolve(self.cfg["api"]["db_path"]))
        self._lock = threading.Lock()
        self.load()

    def load(self) -> bool:
        meta = self.models_dir / "model_meta.json"
        if not meta.exists():
            return False
        self.meta = json.loads(meta.read_text())
        self.models = {k: joblib.load(self.models_dir / f"{k}.joblib") for k in self.meta["models"]
                       if (self.models_dir / f"{k}.joblib").exists()}
        if self.meta["model_key"] not in self.models:
            return False
        self.model = self.models[self.meta["model_key"]]
        sp = self.models_dir / "similarity_index.joblib"
        s = self.cfg["similarity"]
        self.sim = SimilarityIndex(sp, s["near_duplicate"], s["similar"]) if sp.exists() else None
        # Training-support threshold: 1st percentile of word counts seen in training. Inputs shorter than this
        # are out-of-support (almost no training evidence), so we flag uncertainty - we never assign a label by length.
        rs = self.models_dir / "reference_stats.json"
        self.min_support_words = int(np.percentile(json.loads(rs.read_text())["features"]["word_count"], 1)) if rs.exists() else 0
        return True

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _require(self):
        if not self.ready:
            raise ModelNotReady("No trained model found. Run `python -m src.models.train` first.")

    def _level(self, conf: float) -> str:
        c = self.cfg["confidence"]
        return confidence_level(conf, c["high"], c["medium"])

    def _warning(self, label: str, conf: float, level: str, in_support: bool = True):
        if not in_support:
            return (f"Out-of-support input: the training data contains almost no reviews shorter than {self.min_support_words} words, "
                    f"so the model has little real evidence here and its probability ({conf:.1%}) is not reliable. "
                    f"Do not treat this prediction as meaningful.")
        if level == "LOW":
            return (f"Low-confidence prediction ({conf:.1%}). The review does not contain enough strong evidence "
                    f"for a confident classification and should not be treated as definitive.")
        return None

    def pick(self, key: str | None):
        """Resolve a model key (None/'auto' = default). Raises KeyError for unknown keys."""
        key = self.meta["model_key"] if key in (None, "", "auto") else key
        if key not in self.models:
            raise KeyError(key)
        return key, self.models[key]

    def compare(self, text: str) -> list[dict]:
        """Every model's verdict on one review (no explanations, not logged)."""
        self._require()
        out = []
        for k, m in self.models.items():
            p = float(m.predict_proba([text])[:, 1][0])
            lab = "FAKE" if p >= 0.5 else "GENUINE"
            conf = p if lab == "FAKE" else 1 - p
            out.append({"key": k, "model": self.meta["models"][k], "prediction": lab, "confidence": round(conf, 4),
                        "p_fake": round(p, 4), "is_default": k == self.meta["model_key"]})
        return out

    def predict(self, text: str, explain: bool = True, source: str = "single", model: str | None = None) -> dict:
        self._require()
        key, mdl = self.pick(model)
        t0 = time.perf_counter()
        with self._lock:
            p_fake = float(mdl.predict_proba([text])[:, 1][0])
        p_fake = min(max(p_fake, 0.0), 1.0)
        label = "FAKE" if p_fake >= 0.5 else "GENUINE"
        conf = p_fake if label == "FAKE" else 1 - p_fake
        level = self._level(conf)
        in_support = len(text.split()) >= self.min_support_words
        if not in_support:
            level = "LOW"
        st = style_features(text)
        sim = self.sim.query(text) if self.sim else {"max_similarity": 0.0, "level": "NONE", "warning": False,
                                                    "exact_duplicate": False, "closest_text": None}
        contrib, lines = None, []
        if explain:
            contrib = occlusion(mdl, text)
            lines = describe(label, contrib, st, sim)
        ms = (time.perf_counter() - t0) * 1000
        out = {"prediction": label, "confidence": round(conf, 4), "confidence_level": level,
               "probabilities": {"GENUINE": round(1 - p_fake, 4), "FAKE": round(p_fake, 4)},
               "model": self.meta["models"][key], "model_key": key, "processing_time_ms": round(ms, 1),
               "explanation": lines, "contributions": contrib, "warning": self._warning(label, conf, level, in_support), "in_training_support": in_support,
               "similarity_warning": bool(sim["warning"]), "similarity": sim,
               "statistics": {"word_count": len(text.split()), "char_count": len(text),
                              "sentence_count": st["sentence_count"], "sentiment": round(st["sentiment_compound"], 3),
                              "lexical_diversity": round(st["lexical_diversity"], 3)},
               "disclaimer": DISCLAIMER}
        self.store.log([{"text": text, "label": label, "p_fake": p_fake, "confidence": conf, "confidence_level": level,
                         "model": self.meta["models"][key], "latency_ms": ms, "word_count": len(text.split()),
                         "sentiment": st["sentiment_compound"], "similarity": sim["max_similarity"], "source": source}])
        return out

    def predict_batch(self, texts: list[str], explain: bool = False, model: str | None = None) -> dict:
        self._require()
        key, mdl = self.pick(model)
        t0 = time.perf_counter()
        with self._lock:
            p = mdl.predict_proba(texts)[:, 1]
        rows, items = [], []
        for t, pf in zip(texts, p):
            pf = float(min(max(pf, 0.0), 1.0))
            label = "FAKE" if pf >= 0.5 else "GENUINE"
            conf = pf if label == "FAKE" else 1 - pf
            sent = style_features(t)["sentiment_compound"]
            sim = self.sim.query(t)["max_similarity"] if self.sim else 0.0
            lvl = self._level(conf) if len(t.split()) >= self.min_support_words else "LOW"
            item = {"prediction": label, "confidence": round(conf, 4), "confidence_level": lvl,
                    "p_fake": round(pf, 4), "sentiment": round(sent, 3), "similarity_score": sim}
            if explain:
                ex = occlusion(mdl, t, top_k=3)
                side = ex["toward_fake"] if label == "FAKE" else ex["toward_genuine"]
                item["explanation"] = ", ".join(s["token"] for s in side) or None
            items.append(item)
            rows.append({"text": t, "label": label, "p_fake": pf, "confidence": conf,
                         "confidence_level": item["confidence_level"], "model": self.meta["models"][key],
                         "latency_ms": 0.0, "word_count": len(t.split()), "sentiment": sent, "similarity": sim,
                         "source": "batch"})
        total = (time.perf_counter() - t0) * 1000
        for r in rows:
            r["latency_ms"] = total / len(rows)
        self.store.log(rows)
        return {"model": self.meta["models"][key], "count": len(items), "processing_time_ms": round(total, 1),
                "avg_time_per_review_ms": round(total / len(items), 2), "results": items}

    def lime(self, text: str, model: str | None = None) -> dict:
        self._require()
        return lime_explain(self.pick(model)[1], text)

    def report(self) -> dict | None:
        p = self.reports_dir / "evaluation.json"
        return json.loads(p.read_text()) if p.exists() else None
