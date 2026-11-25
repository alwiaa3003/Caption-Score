import numpy as np
import pandas as pd
import pytest

from src.labels import assign_age_bucket, label_top_pct


def test_top_10pct_of_distinct_views_gives_exactly_10pct():
    df = pd.DataFrame({"g": "a", "views": np.arange(1, 101)})
    y = label_top_pct(df, 0.10, ["g"])
    assert y.sum() == 10
    assert set(df.loc[y, "views"]) == set(range(91, 101))


@pytest.mark.parametrize("pct,expected", [(0.05, 5), (0.10, 10), (0.20, 20)])
def test_threshold_is_configurable(pct, expected):
    df = pd.DataFrame({"g": "a", "views": np.arange(100)})
    assert label_top_pct(df, pct, ["g"]).sum() == expected


def test_small_group_rounds_up():
    df = pd.DataFrame({"g": "a", "views": [1, 2, 3]})
    assert label_top_pct(df, 0.10, ["g"]).tolist() == [False, False, True]


def test_ties_at_cutoff_are_all_positive():
    df = pd.DataFrame({"g": "a", "views": [1] * 8 + [5, 5]})
    # k = 1, but both 5s share rank 1
    assert label_top_pct(df, 0.10, ["g"]).tolist() == [False] * 8 + [True, True]


def test_groups_are_labelled_independently():
    # A "small" month must still get its own positives, even though every view count in it
    # is below every view count in the "big" month.
    df = pd.DataFrame({"month": ["small"] * 10 + ["big"] * 10,
                       "views": list(range(10)) + list(range(1000, 1010))})
    y = label_top_pct(df, 0.10, ["month"])
    assert y[df.month == "small"].sum() == 1
    assert y[df.month == "big"].sum() == 1
    assert df.loc[y & (df.month == "small"), "views"].item() == 9


def test_label_preserves_index_alignment():
    df = pd.DataFrame({"g": ["a", "b"] * 5, "views": range(10)}, index=range(100, 110))
    y = label_top_pct(df, 0.10, ["g"])
    assert list(y.sort_index().index) == list(df.index)
    assert df.loc[y.sort_index()[y.sort_index()].index, "views"].tolist() == [8, 9]


def test_invalid_pct_raises():
    df = pd.DataFrame({"g": "a", "views": [1, 2]})
    with pytest.raises(ValueError):
        label_top_pct(df, 1.5, ["g"])


def test_age_buckets_are_equal_size_and_ordered_within_each_month():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"month": ["m1"] * 9 + ["m2"] * 10, "age_days": rng.uniform(30, 900, 19)})
    b = assign_age_bucket(df, 3)
    assert b[df.month == "m1"].value_counts().tolist() == [3, 3, 3]
    assert sorted(b[df.month == "m2"].value_counts().tolist()) == [3, 3, 4]
    for _, g in df.assign(b=b).groupby("month"):
        # every age in bucket i is <= every age in bucket i+1
        assert (g.groupby("b").age_days.max().values[:-1]
                <= g.groupby("b").age_days.min().values[1:]).all()


def test_one_bucket_means_month_only():
    df = pd.DataFrame({"month": ["m"] * 4, "age_days": [1.0, 2.0, 3.0, 4.0]})
    assert assign_age_bucket(df, 1).tolist() == [0, 0, 0, 0]
