"""Phase 2: apply labels, time splits and features; save ml/data/dataset.parquet.

Also writes reports/label_summary.json (class balance per split and per month).

Usage:  python -m src.build_dataset [--config configs/label.yaml]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.cleaning import age_at_capture_days
from src.features import build_features
from src.labels import assign_age_bucket, label_top_pct
from src.splits import SPLIT_ORDER, assign_split

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ML_DIR / "configs" / "label.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    lab = cfg["label"]

    df = pd.read_parquet(REPO_ROOT / cfg["input"])
    n_in = len(df)
    if lab["exclude_ads"]:
        df = df[df["is_ad"] == 0]
    n_ads = n_in - len(df)

    df = df.assign(month=df["posted_at"].dt.strftime("%Y-%m"), age_days=age_at_capture_days(df))
    df["age_bucket"] = assign_age_bucket(df, lab["age_buckets"])
    df["viral"] = label_top_pct(df, lab["top_pct"], ["month", "age_bucket"])
    df["split"] = assign_split(df["month"], cfg["splits"])
    unassigned = int(df["split"].isna().sum())
    df = df[df["split"].notna()]
    df = build_features(df)

    out = REPO_ROOT / cfg["output"]
    df.to_parquet(out, index=False)

    # ---- label summary -------------------------------------------------------------------
    per_split = (df.groupby("split").agg(rows=("viral", "size"), positives=("viral", "sum"),
                                         positive_rate=("viral", "mean"),
                                         months=("month", "nunique"))
                 .reindex(SPLIT_ORDER))
    per_month = df.groupby(["split", "month"]).agg(rows=("viral", "size"),
                                                   positive_rate=("viral", "mean"))
    # Minimum views needed to be "viral" in each month x age bucket (the label's real bar)
    bars = (df[df["viral"]].groupby(["month", "age_bucket"])["views"].min().unstack()
            .add_prefix("min_viral_views_age_bucket_"))
    per_month = per_month.join(bars, on="month")
    per_bucket = df.groupby("age_bucket")["viral"].mean()

    summary = {
        "config": cfg,
        "rows_in": n_in,
        "ads_excluded": n_ads,
        "rows_outside_splits": unassigned,
        "rows_out": len(df),
        "per_split": per_split.round(4).reset_index().to_dict(orient="records"),
        "positive_rate_by_age_bucket": per_bucket.round(4).to_dict(),
        "per_month": per_month.round(4).reset_index().to_dict(orient="records"),
    }
    (ML_DIR / "reports" / "label_summary.json").write_text(json.dumps(summary, indent=2,
                                                                       default=str))
    pd.set_option("display.width", 200)
    print(f"rows in {n_in:,}; ads excluded {n_ads:,}; outside splits {unassigned}; "
          f"rows out {len(df):,}\n")
    print("Per split:\n", per_split.round(4).to_string(), "\n")
    print("Positive rate by age bucket (0 = youngest):", per_bucket.round(4).to_dict(), "\n")
    print("Per month:\n", per_month.round(4).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
