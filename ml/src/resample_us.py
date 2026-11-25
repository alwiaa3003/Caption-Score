"""Step 2 of sampling: make the sample US-only by replacing mixed-country row groups.

Why: each monthly file is sorted by (country, posted_at). Step 1 (src/sample_data.py) drew 2
random row groups per month, and some came from the mixed-country blocks at the start/end of
a file, so those months were ~50% non-US. v1 is US-only.

What it does, per month:
  - reads the file footer and finds the "pure US" row groups (country min == max == "US")
  - keeps every sampled row group that is already pure US (its rows are reused, no download)
  - replaces every other sampled row group with a random pure-US row group from the SAME
    stratum (half of the file) that wasn't already chosen
  - reads the replacements, applies the cleaning rules (incl. country == US) and takes the
    same per-row-group quota as step 1

Running step 1 and then this step with the same config reproduces the final sample exactly.

Usage:
  python -m src.resample_us --dry-run      # footers only, prints replacements and bytes
  python -m src.resample_us --confirm      # run (required above max_download_gb)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfFileSystem

from src.cleaning import clean_mask
from src.sample_data import file_path, raw_stats

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent


def pure_country_row_groups(md, country: str) -> list[int]:
    """Row groups whose country column min and max both equal `country`."""
    ci = next(i for i in range(md.num_columns) if md.schema.column(i).path == "country")
    out = []
    for r in range(md.num_row_groups):
        s = md.row_group(r).column(ci).statistics
        if s is not None and s.has_min_max and s.min == country and s.max == country:
            out.append(r)
    return out


def plan_replacements(num_row_groups: int, k: int, chosen: list[int], pure: list[int],
                      rng: np.random.Generator) -> tuple[list[int], dict[int, int]]:
    """Return (kept row groups, {old_rg: new_rg}) using the same strata as step 1."""
    edges = np.linspace(0, num_row_groups, k + 1).round().astype(int)
    keep, replace = [], {}
    taken = set(chosen)
    for rg in chosen:
        if rg in pure:
            keep.append(rg)
            continue
        s = int(np.searchsorted(edges, rg, side="right") - 1)
        cands = [p for p in pure if edges[s] <= p < edges[s + 1] and p not in taken]
        if not cands:  # stratum has no free pure-US row group: fall back to any free one
            cands = [p for p in pure if p not in taken]
        new = int(rng.choice(cands))
        replace[rg] = new
        taken.add(new)
    return keep, replace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ML_DIR / "configs" / "sample.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    country, k = cfg["country"], cfg["row_groups_per_month"]
    rng = np.random.default_rng(cfg["us_seed"])
    fs = HfFileSystem()
    sample = pd.read_parquet(REPO_ROOT / cfg["output"])
    sample = sample[~sample["source_month"].isin(cfg["drop_months"])]

    # ---- plan (footers only) -------------------------------------------------------------
    plans = []
    for month, rows in sample.groupby("source_month", sort=True):
        chosen = sorted(rows["source_row_group"].unique().tolist())
        with fs.open(file_path(cfg["repo_id"], month), "rb", block_size=1 << 20) as f:
            md = pq.ParquetFile(f).metadata
        pure = pure_country_row_groups(md, country)
        keep, replace = plan_replacements(md.num_row_groups, k, chosen, pure, rng)
        names = {md.schema.column(i).path.split(".")[0]: i for i in range(md.num_columns)}
        nbytes = sum(md.row_group(new).column(names[c]).total_compressed_size
                     for new in replace.values() for c in cfg["columns"])
        plans.append({"month": month, "keep": keep, "replace": replace, "bytes": nbytes})
        if replace:
            print(f"{month}: keep {keep}, replace {replace} -> {nbytes / 1e6:,.0f} MB")
    total_gb = sum(p["bytes"] for p in plans) / 1e9
    n_new = sum(len(p["replace"]) for p in plans)
    print(f"\n{sum(1 for p in plans if p['replace'])} months affected, {n_new} row groups "
          f"to read, TOTAL {total_gb:.2f} GB")
    if args.dry_run:
        return
    if total_gb > cfg["max_download_gb"] and not args.confirm:
        raise SystemExit("Refusing without --confirm.")

    # ---- read replacements and assemble the US-only sample ---------------------------------
    per_rg = int(np.ceil(cfg["rows_per_month"] / k))
    parts, new_rg_stats = [], []
    for p in plans:
        month_rows = sample[(sample["source_month"] == p["month"])
                            & sample["source_row_group"].isin(p["keep"])]
        parts.append(month_rows[month_rows["country"] == country])
        if not p["replace"]:
            continue
        t0 = time.time()
        with fs.open(file_path(cfg["repo_id"], p["month"]), "rb", block_size=8 << 20) as f:
            pf = pq.ParquetFile(f)
            for old, new in p["replace"].items():
                df = pf.read_row_group(new, columns=cfg["columns"]).to_pandas()
                st = raw_stats(df)
                clean = df[clean_mask(df, language=cfg["language"], country=country)]
                n = min(per_rg, len(clean))
                parts.append(clean.sample(n=n, random_state=int(rng.integers(1 << 31)))
                             .assign(source_month=p["month"], source_row_group=new))
                new_rg_stats.append({
                    "month": p["month"], "replaced": old, "new": new, "raw_rows": st["rows"],
                    "raw_en_share": round(st["language"].get(cfg["language"], 0) / st["rows"], 4),
                    "clean_rows_available": len(clean), "kept": n,
                })
        print(f"{p['month']}: replaced {p['replace']} in {time.time() - t0:.0f}s", flush=True)

    us = pd.concat(parts, ignore_index=True).drop_duplicates("video_id")
    out = REPO_ROOT / cfg["us_output"]
    us.to_parquet(out, index=False)
    print(f"\nSaved {len(us):,} US rows to {out}")
    (ML_DIR / "reports" / "resample_us_stats.json").write_text(json.dumps({
        "total_download_gb": round(total_gb, 3),
        "drop_months": cfg["drop_months"],
        "plans": [{**p, "replace": {str(a): b for a, b in p["replace"].items()}} for p in plans],
        "replacement_row_groups": new_rg_stats,
        "rows": len(us),
        "rows_per_month": us["source_month"].value_counts().sort_index().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
