"""Phase 4b step 3: score the DistilBERT validation predictions brought back from Colab.

Put the downloaded files in ml/data/colab/results/:
  val_predictions_distilbert.parquet, run_info.json
  (optional) val_predictions_distilbert_shuffled.parquet  -> shuffled-label sanity check

Usage:  python -m src.import_transformer_preds
Writes: reports/metrics.json ("transformer" section)
"""

import json
from pathlib import Path

import numpy as np

from src.evaluation import evaluate
from src.train_baselines import load, report

ML_DIR = Path(__file__).resolve().parents[1]
RESULTS = ML_DIR / "data" / "colab" / "results"


def scores_for(va, path: Path) -> np.ndarray:
    import pandas as pd

    pred = pd.read_parquet(path)
    merged = va[["video_id"]].merge(pred, on="video_id", how="left", validate="one_to_one")
    if merged["prob"].isna().any():
        raise ValueError(f"{path.name}: {merged['prob'].isna().sum()} val rows have no prediction")
    return merged["prob"].to_numpy()


def main() -> None:
    df = load()
    va = df[df["split"] == "val"].reset_index(drop=True)
    y = va["viral"].to_numpy().astype(int)
    run_info = json.loads((RESULTS / "run_info.json").read_text())
    r = report("DistilBERT", y, scores_for(va, RESULTS / "val_predictions_distilbert.parquet"), va,
               {"run_info": run_info})
    shuffled = RESULTS / "val_predictions_distilbert_shuffled.parquet"
    if shuffled.exists():
        r["sanity_shuffled_labels_val_pr_auc"] = evaluate(y, scores_for(va, shuffled))["pr_auc"]
        print("shuffled-label sanity PR-AUC:", r["sanity_shuffled_labels_val_pr_auc"])

    path = ML_DIR / "reports" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["transformer"] = {"evaluated_on": "validation 2025-09..2026-04 (test untouched)",
                              "distilbert": r}
    path.write_text(json.dumps(metrics, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
