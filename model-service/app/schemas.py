"""Request/response models. Pydantic does all input validation (422 on bad input)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.explain import MAX_CAPTION_CHARS

MAX_HASHTAGS = 30


class PredictRequest(BaseModel):
    caption: str = Field(..., max_length=MAX_CAPTION_CHARS,
                         description=f"Post caption, 1-{MAX_CAPTION_CHARS} characters")
    hashtags: list[str] | None = Field(default=None, max_length=MAX_HASHTAGS,
                                       description="Extra hashtags, with or without '#'")
    post_type: Literal["video", "photo", "reel", "other"] | None = None
    duration_sec: float | None = Field(default=None, gt=0, le=3600)

    @field_validator("caption")
    @classmethod
    def caption_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("caption must not be empty")
        return v

    @field_validator("hashtags")
    @classmethod
    def valid_hashtags(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        for tag in v:
            t = tag.strip().lstrip("#")
            if not t or len(t) > 100 or not t.replace("_", "").isalnum():
                raise ValueError(f"invalid hashtag: {tag!r} (letters, digits and _ only)")
        return v


class TokenWeight(BaseModel):
    token: str
    weight: float = Field(description="contribution to the model's log-odds (+ raises score)")


class HashtagSuggestion(BaseModel):
    hashtag: str
    lift: float = Field(description="viral rate of training posts with this tag / base rate")
    relevance: float
    probability_change: float


class PredictResponse(BaseModel):
    probability: float = Field(description="estimated chance of being in the top 10%")
    label: Literal["likely", "unlikely"]
    top_positive_tokens: list[TokenWeight]
    top_negative_tokens: list[TokenWeight]
    suggested_hashtags: list[HashtagSuggestion]
    model_version: str
    # extras for the UI
    base_rate: float = Field(description="share of posts that are top 10% (~0.10)")
    likely_threshold: float
    tokens: list[TokenWeight] = Field(description="every caption token in order, for highlighting")
    notes: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
