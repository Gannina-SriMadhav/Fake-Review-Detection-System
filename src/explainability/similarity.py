"""Secondary similarity layer (TF-IDF cosine). A supporting signal only - never decides the label."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.cleaning import normalize_text


def build_index(texts: list[str], path: Path) -> None:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform([normalize_text(t) for t in texts]).tocsr()
    joblib.dump({"vec": vec, "X": X, "texts": [normalize_text(t) for t in texts]}, path, compress=3)


class SimilarityIndex:
    def __init__(self, path: Path, near: float = 0.90, similar: float = 0.75):
        d = joblib.load(path)
        self.vec, self.X, self.texts = d["vec"], d["X"], d["texts"]
        self.near, self.similar = near, similar

    def query(self, text: str) -> dict:
        t = normalize_text(text)
        q = self.vec.transform([t])
        if q.nnz == 0:
            return {"max_similarity": 0.0, "level": "NONE", "warning": False, "exact_duplicate": False, "closest_text": None}
        sims = (self.X @ q.T).toarray().ravel()
        i = int(sims.argmax())
        s = float(sims[i])
        exact = self.texts[i].lower() == t.lower()
        level = "EXACT" if exact else "NEAR_DUPLICATE" if s >= self.near else "SIMILAR" if s >= self.similar else "NONE"
        return {"max_similarity": round(min(s, 1.0), 4), "level": level, "warning": level != "NONE",
                "exact_duplicate": exact, "closest_text": self.texts[i][:300] if level != "NONE" else None}
