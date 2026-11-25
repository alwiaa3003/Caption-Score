"""Build a time-spread sample of the TikTok dataset without downloading it.

How the dataset is laid out (verified from the Parquet footers, see docs/data_notes.md):
  - one Parquet file per calendar month (data/YYYY-MM.parquet)
  - each file is sorted by (country, posted_at) and split into row groups of ~1M rows
  - pyarrow can only read whole row groups, so the row group is our smallest unit of download

Sampling design:
  1. For every month in [start_month, end_month] we pick `row_groups_per_month` row groups at
     random (seeded), one from each of k equal blocks of the file (stratified). All row groups
     have ~equal size, so this is an unbiased cluster sample of that month's rows.
  2. We read only the columns listed in the config (column projection) over HTTP range requests.
  3. Before filtering we record raw statistics (language mix, type mix, null rates...), so the
     data notes can describe the data as it is, not just what survives our filters.
  4. We apply the cleaning rules (src/cleaning.py) and take a seeded random subsample so every
     month contributes the same number of rows (rows_per_month).

Usage:
  python -m src.sample_data --dry-run      # read footers only, print the download size
  python -m src.sample_data                # run (refuses above max_download_gb)
  python -m src.sample_data --confirm      # run even above max_download_gb
"""

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfFileSystem

from src.cleaning import clean_mask

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings."""
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def file_path(repo_id: str, month: str) -> str:
    return f"datasets/{repo_id}/data/{month}.parquet"


def plan_month(fs: HfFileSystem, repo_id: str, month: str, columns: list[str], k: int,
               rng: np.random.Generator) -> dict:
    """Read one file's footer and choose which row groups to download."""
    with fs.open(file_path(repo_id, month), "rb", block_size=1 << 20) as f:
        md = pq.ParquetFile(f).metadata

    # Map top-level column name -> indices of its leaf columns (lists have a nested leaf path).
    leaf_idx: dict[str, list[int]] = {}
    for i in range(md.num_columns):
        top = md.schema.column(i).path.split(".")[0]
        leaf_idx.setdefault(top, []).append(i)
    missing = [c for c in columns if c not in leaf_idx]
    if missing:
        raise ValueError(f"{month}: columns not in file: {missing}")

    # Stratified pick: split the row-group index range into k equal blocks and draw one row
    # group at random from each block. Files are sorted by (country, posted_at), so for the big
    # time-sorted US block this spreads picks across the month instead of risking two adjacent
    # days. Every row group still has the same chance of selection (unbiased cluster sample).
    k = min(k, md.num_row_groups)
    edges = np.linspace(0, md.num_row_groups, k + 1).round().astype(int)
    chosen = [int(rng.integers(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    nbytes = 0
    rows = 0
    for rg in chosen:
        g = md.row_group(rg)
        rows += g.num_rows
        nbytes += sum(g.column(i).total_compressed_size for c in columns for i in leaf_idx[c])
    return {"month": month, "file_rows": md.num_rows, "num_row_groups": md.num_row_groups,
            "row_groups": chosen, "rows_to_read": rows, "bytes_to_read": nbytes}


def raw_stats(df: pd.DataFrame) -> dict:
    """Statistics of the raw (unfiltered) rows we read, aggregated later across months."""
    caption = df["caption"].fillna("").str.strip()
    return {
        "rows": len(df),
        "language": Counter(df["language"].fillna("<null>")),
        "type": Counter(df["type"].fillna("<null>")),
        "country": Counter(df["country"].fillna("<null>")),
        "null_counts": df.isna().sum().to_dict(),
        "empty_caption": int((caption == "").sum()),
        "zero_or_null_views": int((df["views"].isna() | (df["views"] == 0)).sum()),
    }


def merge_stats(total: dict, part: dict) -> dict:
    if not total:
        return part
    for key in ("language", "type", "country"):
        total[key] += part[key]
    for key in ("rows", "empty_caption", "zero_or_null_views"):
        total[key] += part[key]
    for col, n in part["null_counts"].items():
        total["null_counts"][col] = total["null_counts"].get(col, 0) + n
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ML_DIR / "configs" / "sample.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="allow download > max_download_gb")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rng = np.random.default_rng(cfg["seed"])
    fs = HfFileSystem()
    months = month_range(cfg["start_month"], cfg["end_month"])

    # ---- 1. Plan: footers only -------------------------------------------------------------
    plans = []
    for m in months:
        plans.append(plan_month(fs, cfg["repo_id"], m, cfg["columns"],
                                cfg["row_groups_per_month"], rng))
        p = plans[-1]
        print(f"{m}: {p['num_row_groups']:3d} row groups, reading {p['row_groups']} "
              f"-> {p['rows_to_read']:,} rows, {p['bytes_to_read'] / 1e6:,.0f} MB")
    total_gb = sum(p["bytes_to_read"] for p in plans) / 1e9
    print(f"\nTOTAL to download: {total_gb:.2f} GB "
          f"({sum(p['rows_to_read'] for p in plans):,} raw rows, {len(months)} months)")
    if args.dry_run:
        return
    if total_gb > cfg["max_download_gb"] and not args.confirm:
        raise SystemExit(f"Refusing: {total_gb:.2f} GB > max_download_gb={cfg['max_download_gb']}."
                         " Re-run with --confirm after approval.")

    # ---- 2. Read, record raw stats, clean, subsample ---------------------------------------
    per_rg_target = int(np.ceil(cfg["rows_per_month"] / cfg["row_groups_per_month"]))
    parts, stats, per_month = [], {}, []
    for p in plans:
        t0 = time.time()
        kept_month = 0
        raw_stats_month = {}
        with fs.open(file_path(cfg["repo_id"], p["month"]), "rb", block_size=8 << 20) as f:
            pf = pq.ParquetFile(f)
            for rg in p["row_groups"]:
                df = pf.read_row_group(rg, columns=cfg["columns"]).to_pandas()
                part = raw_stats(df)
                # deepcopy so the overall and per-month accumulators never share objects
                stats = merge_stats(stats, copy.deepcopy(part))
                raw_stats_month = merge_stats(raw_stats_month, part)
                clean = df[clean_mask(df, language=cfg["language"])]
                n = min(per_rg_target, len(clean))
                sample = clean.sample(n=n, random_state=int(rng.integers(1 << 31)))
                sample = sample.assign(source_month=p["month"], source_row_group=rg)
                parts.append(sample)
                kept_month += n
        lang = raw_stats_month["language"]
        per_month.append({
            "month": p["month"], "row_groups": p["row_groups"],
            "raw_rows": raw_stats_month["rows"], "kept_rows": kept_month,
            "raw_en_share": round(lang.get(cfg["language"], 0) / raw_stats_month["rows"], 4),
        })
        print(f"{p['month']}: kept {kept_month:,} rows in {time.time() - t0:.0f}s", flush=True)

    sample = pd.concat(parts, ignore_index=True)
    before = len(sample)
    sample = sample.drop_duplicates("video_id")
    out = REPO_ROOT / cfg["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False)
    print(f"\nSaved {len(sample):,} rows ({before - len(sample)} duplicate ids dropped) to {out}")

    # ---- 3. Save raw stats for docs/data_notes.md ------------------------------------------
    report = {
        "config": cfg,
        "total_download_gb": round(total_gb, 3),
        "raw": {
            "rows": stats["rows"],
            "language_top30": dict(stats["language"].most_common(30)),
            "type": dict(stats["type"]),
            "country_top30": dict(stats["country"].most_common(30)),
            "null_rate": {c: round(n / stats["rows"], 4) for c, n in stats["null_counts"].items()},
            "empty_caption_rate": round(stats["empty_caption"] / stats["rows"], 4),
            "zero_or_null_views_rate": round(stats["zero_or_null_views"] / stats["rows"], 4),
        },
        "per_month": per_month,
        "sample_rows": len(sample),
        "duplicates_dropped": before - len(sample),
    }
    rep_path = ML_DIR / "reports" / "sampling_stats.json"
    rep_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved raw stats to {rep_path}")


if __name__ == "__main__":
    main()
