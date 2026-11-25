"""Label logic: "viral" = top X% of views within (posting month x age-at-capture bucket).

Why buckets: within one month, videos whose stats were captured sooner after posting have far
more views (month-only labels gave the youngest third ~16% positives and the oldest ~4%).
Comparing each video only with videos of similar age at capture removes that effect.
See docs/label_notes.md.
"""

import math

import pandas as pd


def assign_age_bucket(df: pd.DataFrame, n_buckets: int, age_col: str = "age_days",
                      group_col: str = "month") -> pd.Series:
    """Split each month into `n_buckets` equal-size groups by age at capture (0 = youngest).

    Ranking with method="first" breaks ties by row order, so bucket sizes differ by at most 1.
    """
    if n_buckets == 1:
        return pd.Series(0, index=df.index, dtype="int64")

    def bucket(ages: pd.Series) -> pd.Series:
        ranks = ages.rank(method="first")  # 1..n
        return ((ranks - 1) * n_buckets // len(ages)).astype("int64")

    return df.groupby(group_col, group_keys=False)[age_col].apply(bucket)


def label_top_pct(df: pd.DataFrame, top_pct: float, group_cols: list[str],
                  views_col: str = "views") -> pd.Series:
    """True for rows in the top `top_pct` of views within each group.

    A group of n rows gets k = ceil(top_pct * n) positives. Ties at the cut-off are all
    labelled positive (rank method="min"), so a group can get slightly more than k.
    """
    if not 0 < top_pct < 1:
        raise ValueError("top_pct must be in (0, 1)")

    def top(views: pd.Series) -> pd.Series:
        k = math.ceil(top_pct * len(views))
        return views.rank(method="min", ascending=False) <= k

    return df.groupby(group_cols, group_keys=False)[views_col].apply(top).astype(bool)
