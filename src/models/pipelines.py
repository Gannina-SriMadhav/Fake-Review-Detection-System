"""Model + feature pipeline builders.

Every deployable model is a sklearn-style object with `predict_proba(list[str]) -> (n, 2)`
(columns: GENUINE, FAKE) that takes *raw* review text, so train/serve preprocessing is identical.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from src.features.style import StyleFeatures
from src.preprocessing.cleaning import normalize_text


def _norm(X):
    return [normalize_text(t) for t in X]


class Dense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return sparse.csr_matrix(X) if not sparse.issparse(X) else X


class ColumnPrefix(BaseEstimator, TransformerMixin):
    """Keep the first `n` columns (used so Naive Bayes can ignore the signed style block)."""

    def __init__(self, n: int = 0):
        self.n = n

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return X[:, : self.n]


class FittedUnion(FeatureUnion):
    """FeatureUnion that reports fitted state (sklearn>=1.8 Pipeline checks its last step)."""

    def fit(self, X, y=None, **kw):
        super().fit(X, y, **kw)
        self.is_fitted_ = True
        return self

    def fit_transform(self, X, y=None, **kw):
        out = super().fit_transform(X, y, **kw)
        self.is_fitted_ = True
        return out


def text_union(f: dict, with_style: bool, word_max=None, char: bool = True) -> FeatureUnion:
    parts = [("word", TfidfVectorizer(ngram_range=tuple(f["word_ngram"]), max_features=word_max or f["word_max_features"],
                                      min_df=f["min_df"], sublinear_tf=True, lowercase=True, dtype=np.float32))]
    if char:
        # char n-grams keep punctuation / repeated characters / spelling style that word tokens drop
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(f["char_ngram"]),
                                              max_features=f["char_max_features"], min_df=f["min_df"],
                                              sublinear_tf=True, lowercase=False, dtype=np.float32)))
    if with_style and f.get("use_style_features", True):
        parts.append(("style", Pipeline([("raw", StyleFeatures()), ("scale", StandardScaler()), ("sp", Dense())])))
    return FittedUnion(parts)


def feature_pipeline(f: dict, kind: str) -> Pipeline:
    """kind: 'linear' (word+char+style), 'text' (word+char only), 'tree' (word 20k + style),
    'dense' (SVD of word tfidf + style, for histogram gradient boosting)."""
    norm = ("normalize", FunctionTransformer(_norm, validate=False))
    if kind == "linear":
        return Pipeline([norm, ("union", text_union(f, True))])
    if kind == "text":
        return Pipeline([norm, ("union", text_union(f, False))])
    if kind == "tree":
        return Pipeline([norm, ("union", text_union(f, True, word_max=20000, char=False))])
    if kind == "dense":
        word = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=f["min_df"],
                                                   sublinear_tf=True, dtype=np.float32)),
                         ("svd", TruncatedSVD(150, random_state=0))])
        style = Pipeline([("raw", StyleFeatures()), ("scale", StandardScaler())])
        return Pipeline([norm, ("union", FittedUnion([("svd", word), ("style", style)]))])
    raise ValueError(kind)


def classifier(name: str, seed: int, params: dict | None = None):
    p = params or {}
    if name == "logreg":
        return LogisticRegression(C=p.get("C", 4.0), max_iter=2000, solver="liblinear", random_state=seed)
    if name == "linear_svc":
        return LinearSVC(C=p.get("C", 0.3), max_iter=10000, random_state=seed)
    if name == "naive_bayes":
        return ComplementNB(alpha=p.get("alpha", 0.3))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 300), min_samples_leaf=1, n_jobs=-1,
                                      random_state=seed, max_features="sqrt")
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=p.get("max_iter", 300), learning_rate=0.08, random_state=seed)
    raise ValueError(name)


FEATURE_KIND = {"logreg": "linear", "linear_svc": "linear", "naive_bayes": "text",
                "random_forest": "tree", "gradient_boosting": "dense"}

DISPLAY = {"logreg": "Logistic Regression + TF-IDF", "linear_svc": "Linear SVM + TF-IDF",
           "naive_bayes": "Naive Bayes + TF-IDF", "random_forest": "Random Forest",
           "gradient_boosting": "Gradient Boosting", "ensemble": "Soft-voting Ensemble (LR+SVM+NB)",
           "transformer": "DistilBERT (fine-tuned)"}


def calibrated(base, method: str, cv: int):
    return CalibratedClassifierCV(base, method=method, cv=cv)


class SoftVotingEnsemble:
    """Average calibrated P(FAKE) of fitted member pipelines. Only shipped if it wins on validation."""

    def __init__(self, members: list):
        self.members = members
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


