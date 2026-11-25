"""Phase 4 step 3b: evaluate the CHOSEN model on TEST once, and on STRESS (2026-08) separately.

- The model is the one chosen on validation (metrics.json -> selection.chosen), exactly as
  trained on the training split. Nothing is refit or re-tuned here.
- F1 uses the threshold tuned on validation, unchanged.
- Refuses to run if test results already exist (use --force only to regenerate the SAME
  evaluation after a crash, never to evaluate a different model).

Usage:  python -m src.evaluate_test
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation import evaluate, per_group_pr_auc
from src.select_model import tfidf
from src.train_baselines import MODELS, SEED

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    path = ML_DIR / "reports" / "metrics.json"
    metrics = json.loads(path.read_text())
    if "final" in metrics and not args.force:
        raise SystemExit("Test set already evaluated (metrics.json -> final). Not running again.")
    chosen = metrics["selection"]["chosen"]
    if chosen != "tfidf_lr":
        raise SystemExit(f"chosen model {chosen!r} not wired up here yet")
    threshold = metrics["baselines"]["tfidf_lr"]["val"]["threshold"]

    df = pd.read_parquet(REPO_ROOT / "ml/data/dataset.parquet",
                         columns=["video_id", "split", "month", "text", "viral"],
                         filters=[("split", "in", ["test", "stress"])])
    flags = pd.read_parquet(REPO_ROOT / "ml/data/dedup_flags.parquet",
                            columns=["video_id", "dup_near_in_train"])
    df = df.merge(flags, on="video_id", how="left", validate="one_to_one")
    bundle = joblib.load(MODELS / "baseline_tfidf_lr.joblib")
    df["score"] = bundle["lr"].predict_proba(tfidf(bundle, df["text"]))[:, 1]

    rng = np.random.default_rng(SEED)
    final = {"model": chosen, "threshold_from_validation": threshold,
             "note": "trained on train only; hyperparameters and threshold from validation"}
    for split in ("test", "stress"):
        s = df[df["split"] == split]
        y, sc = s["viral"].to_numpy().astype(int), s["score"].to_numpy()
        keep = ~s["dup_near_in_train"].to_numpy()
        final[split] = {
            "months": sorted(s["month"].unique().tolist()),
            "model": evaluate(y, sc, threshold=threshold),
            "model_deduped": evaluate(y[keep], sc[keep], threshold=threshold),
            "per_month_pr_auc": per_group_pr_auc(y, sc, s["month"]),
            "majority": evaluate(y, np.zeros(len(y)), threshold=0.0),
            "random": evaluate(y, rng.random(len(y)), threshold=0.5),
        }
        m = final[split]["model"]
        print(f"{split}: PR-AUC {m['pr_auc']} (chance {m['positive_rate']}), ROC-AUC "
              f"{m['roc_auc']}, F1 {m['f1']}, P@10% {m['precision_at_top10']} | deduped PR-AUC "
              f"{final[split]['model_deduped']['pr_auc']} | per month "
              f"{final[split]['per_month_pr_auc']}")
    metrics["final"] = final
    path.write_text(json.dumps(metrics, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
