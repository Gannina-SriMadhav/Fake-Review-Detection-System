"""Stylistic / linguistic features.

These are *supporting signals only*: they are fed to the learned model as
numeric columns next to TF-IDF so it can discover their statistical
relationship with the labels. No threshold on any of them is used as a rule.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_WORD = re.compile(r"[A-Za-z0-9']+")
_SENT = re.compile(r"[.!?]+(?:\s|$)")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_URL = re.compile(r"(https?://\S+|www\.\S+|urltoken)", re.I)
_REPEAT_CHAR = re.compile(r"(.)\1{2,}")

FEATURE_NAMES = [
    "word_count", "char_count", "sentence_count", "avg_word_len", "lexical_diversity",
    "repeated_word_ratio", "punct_ratio", "exclamation_ratio", "question_ratio",
    "uppercase_ratio", "digit_ratio", "emoji_count", "url_count", "repeated_char_runs",
    "avg_sentence_len", "sentiment_compound", "sentiment_pos", "sentiment_neg",
]

_sia = None


def _sentiment(text: str):
    global _sia
    if _sia is None:
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            _sia = SentimentIntensityAnalyzer()
        except Exception:  # lexicon unavailable -> neutral zeros, never crash inference
            _sia = False
    if _sia is False:
        return 0.0, 0.0, 0.0
    s = _sia.polarity_scores(text)
    return s["compound"], s["pos"], s["neg"]


def style_features(text: str) -> dict:
    text = text or ""
    words = _WORD.findall(text)
    n_words = len(words)
    n_chars = len(text)
    lower = [w.lower() for w in words]
    uniq = len(set(lower))
    sentences = max(1, len(_SENT.findall(text)) or (1 if text.strip() else 0))
    letters = sum(c.isalpha() for c in text)
    comp, pos, neg = _sentiment(text)
    denom_c = max(n_chars, 1)
    return {
        "word_count": n_words,
        "char_count": n_chars,
        "sentence_count": sentences,
        "avg_word_len": (sum(map(len, words)) / n_words) if n_words else 0.0,
        "lexical_diversity": (uniq / n_words) if n_words else 0.0,
        "repeated_word_ratio": ((n_words - uniq) / n_words) if n_words else 0.0,
        "punct_ratio": sum(c in "!?.,;:-\"'()" for c in text) / denom_c,
        "exclamation_ratio": text.count("!") / denom_c,
        "question_ratio": text.count("?") / denom_c,
        "uppercase_ratio": (sum(c.isupper() for c in text) / letters) if letters else 0.0,
        "digit_ratio": sum(c.isdigit() for c in text) / denom_c,
        "emoji_count": len(_EMOJI.findall(text)),
        "url_count": len(_URL.findall(text)),
        "repeated_char_runs": len(_REPEAT_CHAR.findall(text)),
        "avg_sentence_len": n_words / sentences,
        "sentiment_compound": comp,
        "sentiment_pos": pos,
        "sentiment_neg": neg,
    }


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Stateless transformer -> dense (n, 18) matrix. Scaling is done by the next pipeline step."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            f = style_features(t)
            rows.append([f[k] for k in FEATURE_NAMES])
        arr = np.asarray(rows, dtype=np.float64)
        # log-compress heavy-tailed count features so scaling is stable
        for name in ("word_count", "char_count", "sentence_count", "emoji_count", "url_count",
                     "repeated_char_runs", "avg_sentence_len"):
            j = FEATURE_NAMES.index(name)
            arr[:, j] = np.log1p(arr[:, j])
        return arr

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)

