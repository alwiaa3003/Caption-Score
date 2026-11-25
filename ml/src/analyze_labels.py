"""Evidence for the label definition (docs/label_notes.md). Writes reports/label_analysis.json.

Questions answered:
  1. Does age at capture vary enough within a month to bias a "top 10% of the month" label?
     We label by month only, then look at the viral rate inside within-month age tertiles.
     If the label were age-neutral, every tertile would be close to 10%.
  2. Is 2026-08 different? We compare it with earlier months at the SAME age at capture.

Usage:  python -m src.analyze_labels
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.cleaning import age_at_capture_days

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent


def main() -> None:
    cfg = yaml.safe_load((ML_DIR / "configs" / "sample.yaml").read_text())
    df = pd.read_parquet(REPO_ROOT / cfg["us_output"])
    df["age"] = age_at_capture_days(df)
    df["month"] = df["posted_at"].dt.strftime("%Y-%m")
    df["log_views"] = np.log10(df["views"].astype(float))
    out: dict = {"rows": len(df), "us_share": float((df["country"] == "US").mean())}

    # ---- 1. month-only label vs within-month age tertiles ----------------------------------
    df["viral_month_only"] = df["views"] >= df.groupby("month")["views"].transform(
        lambda v: v.quantile(0.9))
    df["age_tertile"] = df.groupby("month")["age"].transform(
        lambda a: pd.qcut(a.rank(method="first"), 3, labels=False))
    by_tertile = df.groupby(["month", "age_tertile"])["viral_month_only"].mean().unstack()
    out["month_only_viral_rate_by_age_tertile"] = {
        "mean_over_months": by_tertile.mean().round(4).to_dict(),
        "max_abs_deviation_from_10pct": float((by_tertile - 0.1).abs().max().max().round(4)),
        "months_with_a_tertile_outside_7_to_13pct": int(
            ((by_tertile < 0.07) | (by_tertile > 0.13)).any(axis=1).sum()),
        "per_month": by_tertile.round(3).to_dict(orient="index"),
    }
    young = df[df["age"] < 90]
    out["young_lt_90d"] = {
        "rows": len(young),
        "months": sorted(young["month"].unique().tolist()),
        "spearman_age_logviews": float(young[["age", "log_views"]].corr("spearman").iloc[0, 1]),
    }
    old = df[df["age"] >= 365]
    out["old_ge_365d_spearman_age_logviews"] = float(
        old[["age", "log_views"]].corr("spearman").iloc[0, 1])

    # ---- 2. 2026-08 vs earlier months at matched age (30-63 days, 2026-08's age range) -----
    matched = df[(df["age"] >= 30) & (df["age"] <= 63)]
    qs = [0.5, 0.75, 0.9, 0.99]
    cmp = matched.groupby("month")["views"].quantile(qs).unstack()
    cmp["rows"] = matched.groupby("month").size()
    out["matched_age_30_63d_view_quantiles"] = cmp.round(0).to_dict(orient="index")
    aug = df[df["month"] == "2026-08"]
    out["2026_08_by_row_group"] = (
        aug.groupby("source_row_group")
        .agg(rows=("views", "size"), posted_min=("posted_at", "min"),
             posted_max=("posted_at", "max"), views_median=("views", "median"),
             views_p90=("views", lambda v: v.quantile(0.9)), age_median=("age", "median"),
             is_ad=("is_ad", "mean"))
        .astype(str).to_dict(orient="index"))
    month_q = df.groupby("month")["views"].quantile([0.5, 0.9]).unstack()
    month_q["p90_over_median"] = month_q[0.9] / month_q[0.5]
    out["p90_over_median_by_month"] = month_q["p90_over_median"].round(2).to_dict()

    path = ML_DIR / "reports" / "label_analysis.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("p90_over_median_by_month",)}, indent=1, default=str)[:6000])
    print("\np90/median by month:", out["p90_over_median_by_month"])


if __name__ == "__main__":
    main()
