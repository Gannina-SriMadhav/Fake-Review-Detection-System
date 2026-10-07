"""Dataset loading, validation, de-duplication and leakage-safe splitting."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from src.config import resolve
from src.preprocessing.cleaning import normalize_text

log = logging.getLogger(__name__)


class SchemaError(ValueError):
    pass


def validate_schema(df: pd.DataFrame, text_col: str, label_col: str | None = None) -> None:
    missing = [c for c in [text_col, label_col] if c and c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {missing}. Found: {list(df.columns)}")


def near_duplicate_groups(texts: pd.Series, threshold: float = 0.9, chunk: int = 500) -> np.ndarray:
    """Group reviews whose TF-IDF cosine similarity >= threshold (connected components).

    Used only to keep near-duplicates on the *same side* of a split. The vectoriser here
    is label-free and is not reused as a model feature extractor.
    """
    n = len(texts)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.3, sublinear_tf=True, dtype=np.float32)
    try:
        X = vec.fit_transform(texts.values)
    except ValueError:  # tiny corpora
        return np.arange(n)
    rows, cols = [], []
    XT = X.T.tocsc()
    for s in range(0, n, chunk):
        sim = (X[s:s + chunk] @ XT).tocoo()
        keep = (sim.data >= threshold) & (sim.row + s < sim.col)
        rows.append(sim.row[keep] + s)
        cols.append(sim.col[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Returns (clean dataframe with columns text, label, group, category?) and a data-quality report."""
    d = cfg["data"]
    raw = pd.read_csv(resolve(d["path"]))
    validate_schema(raw, d["text_col"], d["label_col"])
    report: dict = {"raw_rows": int(len(raw))}

    df = raw.rename(columns={d["text_col"]: "text", d["label_col"]: "label_raw"}).copy()
    df["label"] = df["label_raw"].map(d["label_map"])
    report["unmapped_labels"] = int(df["label"].isna().sum())
    report["missing_text"] = int(df["text"].isna().sum())
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].map(normalize_text)
    df = df[(df["text"].str.len() >= d["min_chars"])]
    df["text"] = df["text"].str.slice(0, d["max_chars"])
    df["label"] = df["label"].astype(int)

    # exact duplicates (case-insensitive). Conflicting-label duplicates are ambiguous -> drop all copies.
    key = df["text"].str.lower()
    nlab = df.groupby(key)["label"].transform("nunique")
    report["conflicting_label_duplicates"] = int((nlab > 1).sum())
    df = df[nlab == 1]
    before = len(df)
    df = df[~key.loc[df.index].duplicated(keep="first")]
    report["exact_duplicates_removed"] = int(before - len(df))

    if d.get("sample_size"):
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), d["sample_size"] // 2), random_state=cfg["project"]["seed"]))
    df = df.reset_index(drop=True)

    df["group"] = near_duplicate_groups(df["text"], d["near_dup_threshold"])
    report["near_duplicate_clusters_gt1"] = int((pd.Series(df["group"]).value_counts() > 1).sum())
    report["rows_in_near_dup_clusters"] = int((df.groupby("group")["text"].transform("size") > 1).sum())
    report["clean_rows"] = int(len(df))
    vc = df["label"].value_counts(normalize=True)
    report["fake_fraction"] = float(vc.get(1, 0.0))
    report["label_counts"] = {"GENUINE": int((df.label == 0).sum()), "FAKE": int((df.label == 1).sum())}
    return df, report


def split_dataset(df: pd.DataFrame, cfg: dict):
    """Stratified, group-aware train/val/test split. Near-duplicate clusters never straddle splits."""
    seed = cfg["project"]["seed"]
    d = cfg["data"]

    def carve(frame, frac):
        k = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        a, b = next(sgkf.split(frame, frame["label"], frame["group"]))
        return frame.iloc[a], frame.iloc[b]

    trainval, test = carve(df, d["test_size"])
    train, val = carve(trainval, d["val_size"] / (1 - d["test_size"]))
    for name, part in (("train", train), ("val", val), ("test", test)):
        log.info("%s: %d rows, fake=%.3f", name, len(part), part.label.mean())
    # leakage assertions
    gs = [set(p["group"]) for p in (train, val, test)]
    assert not (gs[0] & gs[1] or gs[0] & gs[2] or gs[1] & gs[2]), "group leakage between splits"
    low = [set(p["text"].str.lower()) for p in (train, val, test)]
    assert not (low[0] & low[1] or low[0] & low[2] or low[1] & low[2]), "exact duplicate leakage"
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
