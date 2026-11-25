"""Profile the cleaned sample: the numbers that go into docs/data_notes.md.

Reads the sample written by src/sample_data.py and saves reports/sample_profile.json.
Raw captions are never printed here (dataset ethics: see README).

Usage:  python -m src.describe_sample
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from src.cleaning import age_at_capture_days

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
HASHTAG_RE = r"#\w+"


def q(series: pd.Series, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)) -> dict:
    return {str(k): round(float(v), 1) for k, v in series.quantile(list(qs)).items()}


def main() -> None:
    cfg = yaml.safe_load((ML_DIR / "configs" / "sample.yaml").read_text())
    df = pd.read_parquet(REPO_ROOT / cfg["output"])
    df["age_days"] = age_at_capture_days(df)
    df["month"] = df["posted_at"].dt.strftime("%Y-%m")
    caption = df["caption"].fillna("")
    n_tags = caption.str.count(HASHTAG_RE)

    per_month = (
        df.groupby("month")
        .agg(
            rows=("video_id", "size"),
            views_median=("views", "median"),
            views_p90=("views", lambda s: s.quantile(0.9)),
            age_min=("age_days", "min"),
            age_median=("age_days", "median"),
            age_max=("age_days", "max"),
        )
        .round(1)
        .reset_index()
        .to_dict(orient="records")
    )

    profile = {
        "rows": len(df),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "posted_at_range": [str(df["posted_at"].min()), str(df["posted_at"].max())],
        "stats_updated_at_range": [str(df["stats_updated_at"].min()),
                                   str(df["stats_updated_at"].max())],
        "duplicate_video_ids": int(df["video_id"].duplicated().sum()),
        "null_rate": df.isna().mean().round(4).to_dict(),
        "country_top15_share": df["country"].value_counts(normalize=True).head(15)
        .round(4).to_dict(),
        "is_ad_rate": round(float((df["is_ad"] == 1).mean()), 4),
        "branded_content": df["branded_content"].value_counts(normalize=True).round(4).to_dict(),
        "not_recommended_for_fyp_rate": round(
            float((df["not_recommended_for_fyp"] == 1).mean()), 4
        ),
        "duration_sec_quantiles": q(df["duration_ms"] / 1000),
        "duration_zero_rate": round(float((df["duration_ms"] == 0).mean()), 4),
        "caption_chars_quantiles": q(caption.str.len()),
        "hashtags_per_caption_quantiles": q(n_tags),
        "share_with_hashtags": round(float((n_tags > 0).mean()), 4),
        "views_quantiles": q(df["views"]),
        "age_at_capture_days_quantiles": q(df["age_days"]),
        "per_month": per_month,
    }
    out = ML_DIR / "reports" / "sample_profile.json"
    out.write_text(json.dumps(profile, indent=2, default=str))
    print(json.dumps({k: v for k, v in profile.items() if k != "per_month"}, indent=2,
                     default=str))
    print("\nper month:")
    print(pd.DataFrame(per_month).to_string(index=False))


if __name__ == "__main__":
    main()
