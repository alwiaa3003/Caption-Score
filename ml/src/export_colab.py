"""Phase 4b step 1: export the data subset for fine-tuning DistilBERT on Google Colab.

Contents of ml/data/colab/captionscore_finetune.parquet:
  - a stratified-by-month subsample of TRAIN (same fraction from every month, seed 42)
  - ALL of validation
  Test and stress are NOT exported.
Columns: video_id, split, month, text (cleaned caption), viral (0/1).

Upload to Colab: this parquet + ml/src/finetune_transformer.py (+ the notebook).
Usage:  python -m src.export_colab [--train-rows 250000]
"""

import argparse
from pathlib import Path

import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
OUT = REPO_ROOT / "ml" / "data" / "colab" / "captionscore_finetune.parquet"


def stratified_by_month(train: pd.DataFrame, n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Take the same fraction of rows from every month (so each month keeps its share)."""
    frac = min(1.0, n_rows / len(train))
    return train.groupby("month", group_keys=False).sample(frac=frac, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=250_000)
    args = parser.parse_args()

    df = pd.read_parquet(REPO_ROOT / "ml/data/dataset.parquet",
                         columns=["video_id", "split", "month", "text", "viral"],
                         filters=[("split", "in", ["train", "val"])])
    assert set(df["split"]) == {"train", "val"}
    sub = stratified_by_month(df[df["split"] == "train"], args.train_rows)
    out = pd.concat([sub, df[df["split"] == "val"]], ignore_index=True)
    out["viral"] = out["viral"].astype("int8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    per_month = sub.groupby("month").size()
    print(f"train subsample: {len(sub):,} rows from {sub['month'].nunique()} months "
          f"({per_month.min():,}-{per_month.max():,} per month), positive rate "
          f"{sub['viral'].mean():.4f}")
    print(f"val: {(out['split'] == 'val').sum():,} rows")
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
