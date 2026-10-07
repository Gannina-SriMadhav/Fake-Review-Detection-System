"""SQLite prediction log (keeps the project's original SQLite persistence)."""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

_SCHEMA = """CREATE TABLE IF NOT EXISTS predictions(
 id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, text TEXT, text_hash TEXT, label TEXT, p_fake REAL,
 confidence REAL, confidence_level TEXT, model TEXT, latency_ms REAL, word_count INTEGER,
 sentiment REAL, similarity REAL, source TEXT)"""


class PredictionStore:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self._lock = str(path), threading.Lock()
        with self._conn() as c:
            c.execute(_SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def log(self, rows: list[dict]) -> None:
        with self._lock, self._conn() as c:
            c.executemany(
                "INSERT INTO predictions(ts,text,text_hash,label,p_fake,confidence,confidence_level,model,latency_ms,"
                "word_count,sentiment,similarity,source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(time.time(), r["text"][:5000], hashlib.sha1(r["text"].lower().encode()).hexdigest(), r["label"],
                  r["p_fake"], r["confidence"], r["confidence_level"], r["model"], r["latency_ms"],
                  r["word_count"], r["sentiment"], r["similarity"], r.get("source", "single")) for r in rows])

    def frame(self, limit: int = 20000) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", c, params=(limit,))

    def seen_hash(self, text: str) -> int:
        h = hashlib.sha1(text.lower().encode()).hexdigest()
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM predictions WHERE text_hash=?", (h,)).fetchone()[0]
