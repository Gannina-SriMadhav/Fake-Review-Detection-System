from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

MAX_CHARS = 5000


class ReviewRequest(BaseModel):
    review: str = Field(..., description="Review text to analyse")
    explain: bool = True
    model: Optional[str] = Field(None, description="Model key, or omit/'auto' for the default")

    @field_validator("review")
    @classmethod
    def _check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Review must not be empty.")
        if len(v) > MAX_CHARS:
            raise ValueError(f"Review exceeds the {MAX_CHARS}-character limit.")
        return v


class BatchRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1)
    explain: bool = False
    model: Optional[str] = None

    @field_validator("reviews")
    @classmethod
    def _check(cls, v: list[str]) -> list[str]:
        out = []
        for i, r in enumerate(v):
            r = (r or "").strip()
            if not r:
                raise ValueError(f"Review at index {i} is empty.")
            out.append(r[:MAX_CHARS])
        return out


class Stats(BaseModel):
    word_count: int
    char_count: int
    sentence_count: int
    sentiment: float
    lexical_diversity: float


class Similarity(BaseModel):
    max_similarity: float
    level: str
    warning: bool
    exact_duplicate: bool = False
    closest_text: Optional[str] = None


class Prediction(BaseModel):
    prediction: str
    confidence: float
    confidence_level: str
    probabilities: dict[str, float]
    model: str
    model_key: str = ""
    processing_time_ms: float
    explanation: list[str] = []
    contributions: Optional[dict] = None
    warning: Optional[str] = None
    in_training_support: bool = True
    similarity_warning: bool = False
    similarity: Optional[Similarity] = None
    statistics: Optional[Stats] = None
    disclaimer: str = ("Model prediction only - an automated estimate, not proof that a review is fraudulent.")


class BatchItem(BaseModel):
    prediction: str
    confidence: float
    confidence_level: str
    p_fake: float
    sentiment: float
    similarity_score: float
    explanation: Optional[str] = None


class BatchResponse(BaseModel):
    model: str
    count: int
    processing_time_ms: float
    avg_time_per_review_ms: float
    results: list[BatchItem]
