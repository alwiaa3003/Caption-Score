from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.splits import assign_split, validate_split_config

SPLITS = {"train": ["2022-01", "2025-08"], "val": ["2025-09", "2026-04"],
          "test": ["2026-05", "2026-07"], "stress": ["2026-08", "2026-08"]}


def all_months():
    months = pd.period_range("2022-01", "2026-08", freq="M")
    return pd.Series([p.strftime("%Y-%m") for p in months])


def test_every_month_gets_exactly_one_split():
    s = assign_split(all_months(), SPLITS)
    assert s.notna().all()
    assert s.value_counts().to_dict() == {"train": 44, "val": 8, "test": 3, "stress": 1}


def test_splits_are_strictly_ordered_in_time():
    months = all_months()
    s = assign_split(months, SPLITS)
    p = pd.PeriodIndex(months, freq="M")
    order = ["train", "val", "test", "stress"]
    for a, b in zip(order, order[1:], strict=False):
        assert p[(s == a).values].max() < p[(s == b).values].min()


def test_boundaries_are_inclusive():
    s = assign_split(pd.Series(["2025-08", "2025-09", "2026-07", "2026-08"]), SPLITS)
    assert s.tolist() == ["train", "val", "test", "stress"]


def test_months_outside_ranges_are_unassigned():
    s = assign_split(pd.Series(["2021-12", "2026-09"]), SPLITS)
    assert s.isna().all()


def test_overlapping_config_is_rejected():
    bad = {**SPLITS, "val": ["2025-08", "2026-04"]}
    with pytest.raises(ValueError):
        validate_split_config(bad)


def test_out_of_order_config_is_rejected():
    bad = {**SPLITS, "test": ["2025-01", "2025-02"]}
    with pytest.raises(ValueError):
        validate_split_config(bad)


def test_project_config_matches_agreed_splits():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "label.yaml").read_text())
    assert cfg["splits"] == SPLITS
