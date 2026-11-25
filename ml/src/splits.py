"""Time-based split by posting month. No shuffling across time."""

import pandas as pd

SPLIT_ORDER = ["train", "val", "test", "stress"]


def validate_split_config(splits: dict[str, list[str]]) -> None:
    """Ranges must be in train < val < test < stress order and must not overlap."""
    prev_end = None
    for name in SPLIT_ORDER:
        if name not in splits:
            continue
        start, end = (pd.Period(m, "M") for m in splits[name])
        if start > end:
            raise ValueError(f"{name}: start {start} is after end {end}")
        if prev_end is not None and start <= prev_end:
            raise ValueError(f"{name} starts at {start}, not after the previous split ({prev_end})")
        prev_end = end


def assign_split(months: pd.Series, splits: dict[str, list[str]]) -> pd.Series:
    """Map 'YYYY-MM' strings to a split name. Months outside every range get None."""
    validate_split_config(splits)
    periods = pd.PeriodIndex(months, freq="M")
    out = pd.Series(None, index=months.index, dtype="object")
    for name, (start, end) in splits.items():
        inside = (periods >= pd.Period(start, "M")) & (periods <= pd.Period(end, "M"))
        out[inside] = name
    return out
