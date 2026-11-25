import pandas as pd

from src.cleaning import age_at_capture_days, clean_mask


def make_df(**overrides):
    """One valid row; each test breaks exactly one rule."""
    row = {
        "language": "en",
        "type": "video",
        "caption": "hello world #fyp",
        "views": 100,
        "posted_at": pd.Timestamp("2024-01-01", tz="UTC"),
        "stats_updated_at": pd.Timestamp("2024-03-01", tz="UTC"),
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_valid_row_is_kept():
    assert clean_mask(make_df()).tolist() == [True]


def test_each_rule_drops_the_row():
    bad = [
        {"language": "es"},
        {"language": "un"},
        {"type": "photo"},
        {"caption": "   "},
        {"caption": None},
        {"views": 0},
        {"views": None},
        {"stats_updated_at": pd.Timestamp("2024-01-20", tz="UTC")},  # 19 days old
    ]
    for override in bad:
        assert clean_mask(make_df(**override)).tolist() == [False], override


def test_age_boundary_is_inclusive_at_30_days():
    df = make_df(stats_updated_at=pd.Timestamp("2024-01-31", tz="UTC"))
    assert age_at_capture_days(df).iloc[0] == 30.0
    assert clean_mask(df).tolist() == [True]
