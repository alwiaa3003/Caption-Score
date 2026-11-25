"""Row-level cleaning rules shared by sampling and (later) label building.

Kept in one place so the rules are tested once and applied identically everywhere.
"""

import pandas as pd

MIN_AGE_DAYS = 30  # videos younger than this at stats capture haven't finished accumulating views


def age_at_capture_days(df: pd.DataFrame) -> pd.Series:
    """Days between posting and when the view count was captured (stats_updated_at)."""
    return (df["stats_updated_at"] - df["posted_at"]).dt.total_seconds() / 86400.0


def clean_mask(
    df: pd.DataFrame, language: str = "en", min_age_days: float = MIN_AGE_DAYS
) -> pd.Series:
    """Boolean mask of rows we keep.

    Rules:
      - language == `language` (v1 is English only)
      - type == "video" (photo carousels are a different format)
      - caption is non-empty after stripping whitespace
      - views is present and > 0
      - age at capture >= min_age_days
    """
    caption = df["caption"].fillna("").str.strip()
    views = df["views"]
    age = age_at_capture_days(df)
    return (
        (df["language"] == language)
        & (df["type"] == "video")
        & (caption != "")
        & views.notna()
        & (views > 0)
        & (age >= min_age_days)
    )
