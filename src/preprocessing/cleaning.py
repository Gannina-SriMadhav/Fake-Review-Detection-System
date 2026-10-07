"""Text normalisation.

Deliberately *light*: punctuation, casing, repeated characters and emoji carry
stylistic signal, so we keep the original text and derive a normalised view
instead of destroying information. No rule here ever decides a label.
"""
from __future__ import annotations

import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")


def basic_clean(text) -> str:
    """Unicode-normalise, strip HTML/control chars, collapse whitespace. Keeps case & punctuation."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = html.unescape(str(text))
    t = unicodedata.normalize("NFKC", t)
    t = _TAG.sub(" ", t)
    t = _CTRL.sub(" ", t)
    return _WS.sub(" ", t).strip()


def normalize_text(text) -> str:
    """Model-facing view: basic_clean + URLs replaced by a token (URL presence stays visible)."""
    return _URL.sub(" urltoken ", basic_clean(text))


def normalize_series(series):
    return series.map(normalize_text)
