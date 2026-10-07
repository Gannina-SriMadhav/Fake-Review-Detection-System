"""Lightweight English-vs-other detection (no external dependency).

The models were trained on English reviews only, so for other languages their output is
meaningless. This does not decide any label - it only tells the app to warn and lower confidence.
"""
from __future__ import annotations

import re

_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
# very common English function words; real English text of 5+ words almost always contains several
_EN = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "it", "this", "that", "i", "my", "me", "we", "you",
       "to", "of", "in", "on", "for", "with", "as", "at", "by", "be", "have", "has", "had", "not", "so", "very", "good", "great",
       "love", "like", "would", "will", "can", "do", "does", "did", "its", "they", "them", "from", "just", "too", "all", "one",
       "product", "great", "nice", "bad", "well", "works", "work", "bought", "buy", "no", "if", "up", "out", "what", "which"}


def detect_language(text: str) -> dict:
    """Returns {supported: bool, reason: str|None}. Unknown/short text is treated as supported (no evidence either way)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {"supported": True, "reason": None}
    latin = sum("LATIN" in _name(c) for c in letters[:400])
    if latin / min(len(letters), 400) < 0.6:
        return {"supported": False, "reason": "non-Latin script"}
    words = [w.lower() for w in _WORDS.findall(text)]
    if len(words) >= 5:
        hit = sum(w in _EN for w in words) / len(words)
        if hit < 0.12:
            return {"supported": False, "reason": "text does not look like English"}
    return {"supported": True, "reason": None}


def _name(c: str) -> str:
    import unicodedata
    try:
        return unicodedata.name(c)
    except ValueError:
        return ""
