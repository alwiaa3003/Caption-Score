"""Phase 4a: classifiers on all-MiniLM-L6-v2 caption embeddings, evaluated on VALIDATION only.

Models
  (e) embeddings               -> Logistic Regression (C tuned on val)
  (f) embeddings + meta        -> LightGBM (num_leaves tuned on val; early stopping on the
                                  last 3 training months, same rule as Phase 3)
Plus: duplicates-removed validation scores, per-month PR-AUC, shuffled-label sanity check.

Usage:  python -m src.train_embeddings    (after python -m src.embed --splits train val)
Writes: reports/metrics.json ("embeddings" section), models/emb_*.{joblib,txt}
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.embed import MODEL, load_embeddings
from src.evaluation import evaluate
from src.features import META_COLUMNS
from src.train_baselines import INNER_MONTHS, MODELS, SEED, lgb_fit, load, log, report

ML_DIR = Path(__file__).resolve().parents[1]


def aligned(split_df, split: str) -> np.ndarray:
    ids, emb = load_embeddings(split)
    if not np.array_equal(ids, split_df["video_id"].to_numpy()):
        raise ValueError(f"{split}: embedding order does not match dataset order")
    return emb


def main() -> None:
    df = load()
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    y_tr, y_va = tr["viral"].to_numpy().astype(int), va["viral"].to_numpy().astype(int)
    inner = tr["month"].isin(INNER_MONTHS).to_numpy()
    e_tr, e_va = aligned(tr, "train"), aligned(va, "val")
    log(f"embeddings train {e_tr.shape}, val {e_va.shape}")
    results: dict = {"encoder": MODEL, "max_seq_length": 64, "rows_encoded": "all train + val"}

    # ---- (e) embeddings -> LR ------------------------------------------------------------------
    best = None
    for c in (0.01, 0.1, 1.0):
        t0 = time.time()
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=c, max_iter=2000, random_state=SEED))
        m.fit(e_tr, y_tr)
        r = report(f"emb LR C={c} ({time.time() - t0:.0f}s)", y_va, m.predict_proba(e_va)[:, 1],
                   va, {"params": {"C": c}})
        if best is None or r["val"]["pr_auc"] > best[0]["val"]["pr_auc"]:
            best = (r, m)
    results["emb_lr"], lr_best = best
    joblib.dump(lr_best, MODELS / "emb_lr.joblib")

    # ---- (f) embeddings + meta -> LightGBM ----------------------------------------------------
    x_tr = np.hstack([e_tr, tr[META_COLUMNS].to_numpy(np.float32)])
    x_va = np.hstack([e_va, va[META_COLUMNS].to_numpy(np.float32)])
    best = None
    for leaves in (63, 255):
        t0 = time.time()
        params = {"num_leaves": leaves, "learning_rate": 0.05, "feature_fraction": 0.5}
        bst = lgb_fit(x_tr[~inner], y_tr[~inner], x_tr[inner], y_tr[inner], params)
        r = report(f"emb+meta LGBM leaves={leaves} ({time.time() - t0:.0f}s)", y_va,
                   bst.predict(x_va), va,
                   {"params": {**params, "best_iteration": bst.best_iteration},
                    "lgb_params": params})
        if best is None or r["val"]["pr_auc"] > best[0]["val"]["pr_auc"]:
            best = (r, bst)
    results["emb_meta_lgbm"], bst_best = best
    bst_best.save_model(str(MODELS / "emb_meta_lgbm.txt"))

    # ---- sanity: shuffled training labels -------------------------------------------------------
    y_shuf = np.random.default_rng(SEED).permutation(y_tr)
    m = make_pipeline(StandardScaler(), LogisticRegression(
        C=results["emb_lr"]["params"]["C"], max_iter=2000, random_state=SEED)).fit(e_tr, y_shuf)
    b = lgb_fit(x_tr[~inner], y_shuf[~inner], x_tr[inner], y_shuf[inner],
                results["emb_meta_lgbm"]["lgb_params"])
    results["sanity_shuffled_labels"] = {
        "emb_lr_val_pr_auc": evaluate(y_va, m.predict_proba(e_va)[:, 1])["pr_auc"],
        "emb_meta_lgbm_val_pr_auc": evaluate(y_va, b.predict(x_va))["pr_auc"],
    }
    log(f"shuffled-label sanity: {results['sanity_shuffled_labels']}")

    path = ML_DIR / "reports" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["embeddings"] = {"evaluated_on": "validation 2025-09..2026-04 (test untouched)",
                             **results}
    path.write_text(json.dumps(metrics, indent=2))
    log(f"saved {path}")


if __name__ == "__main__":
    main()
