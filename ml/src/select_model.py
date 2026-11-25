"""Phase 4 step 3a: choose the final model on VALIDATION only.

Decision rule (fixed before looking at the bootstrap results):
  1. The best model is the one with the highest validation PR-AUC.
  2. TF-IDF + LR is preferred for the app (it gives exact per-word contributions) if it is
     "within noise" of the best: the 95% CI of PR-AUC(best) - PR-AUC(TF-IDF LR) from a paired
     **day-block** bootstrap includes 0. Blocks are posting days, because rows from the same
     day come from the same Parquet row group and are correlated, so a row-level bootstrap
     would understate the noise. The row-level CI is reported too, for comparison.

Also recomputes every saved model's validation scores (cached in data/preds/) and draws the
validation PR curves.

Usage:  python -m src.select_model
Writes: reports/metrics.json ("selection"), reports/val_pr_curves.png, data/preds/val_scores.parquet
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.embed import load_embeddings
from src.features import META_COLUMNS
from src.train_baselines import MODELS, SEED, load, log

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ML_DIR = Path(__file__).resolve().parents[1]
PREDS = ML_DIR / "data" / "preds"
N_BOOT = 1000
NAMES = {"meta_lgbm": "(b) Meta only, LightGBM", "tfidf_lr": "(c) TF-IDF, LR",
         "tfidf_meta_lgbm": "(d) TF-IDF + meta, LightGBM",
         "emb_meta_lgbm": "(f) MiniLM emb. + meta, LightGBM"}


def tfidf(bundle, texts) -> sp.csr_matrix:
    return sp.hstack([bundle["word"].transform(texts), bundle["char"].transform(texts)]).tocsr()


def val_scores(va: pd.DataFrame) -> pd.DataFrame:
    """Scores of each saved model on validation (recomputed from the saved artifacts)."""
    meta = va[META_COLUMNS].to_numpy(np.float32)
    bundle = joblib.load(MODELS / "baseline_tfidf_lr.joblib")
    x = tfidf(bundle, va["text"])
    sel = joblib.load(MODELS / "baseline_tfidf_meta_selector.joblib")
    ids, emb = load_embeddings("val")
    assert np.array_equal(ids, va["video_id"].to_numpy())
    return pd.DataFrame({
        "video_id": va["video_id"].to_numpy(),
        "meta_lgbm": lgb.Booster(model_file=str(MODELS / "baseline_meta_lgbm.txt")).predict(meta),
        "tfidf_lr": bundle["lr"].predict_proba(x)[:, 1],
        "tfidf_meta_lgbm": lgb.Booster(model_file=str(MODELS / "baseline_tfidf_meta_lgbm.txt"))
        .predict(sp.hstack([sel.transform(x), sp.csr_matrix(meta)]).tocsr()),
        "emb_meta_lgbm": lgb.Booster(model_file=str(MODELS / "emb_meta_lgbm.txt"))
        .predict(np.hstack([emb, meta])),
    })


def paired_bootstrap(y, s_a, s_b, blocks, rng) -> dict:
    """PR-AUC(a) - PR-AUC(b): point estimate and 95% CIs (row-level and block-level)."""
    diff = average_precision_score(y, s_a) - average_precision_score(y, s_b)
    n = len(y)
    row = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        row.append(average_precision_score(y[i], s_a[i]) - average_precision_score(y[i], s_b[i]))
    uniq = np.unique(blocks)
    idx_by_block = {b: np.flatnonzero(blocks == b) for b in uniq}
    blk = []
    for _ in range(N_BOOT):
        i = np.concatenate([idx_by_block[b] for b in rng.choice(uniq, len(uniq))])
        blk.append(average_precision_score(y[i], s_a[i]) - average_precision_score(y[i], s_b[i]))
    return {"diff": round(float(diff), 4),
            "row_ci95": [round(float(np.percentile(row, q)), 4) for q in (2.5, 97.5)],
            "day_block_ci95": [round(float(np.percentile(blk, q)), 4) for q in (2.5, 97.5)],
            "n_day_blocks": int(len(uniq))}


def main() -> None:
    df = load()
    va = df[df["split"] == "val"].reset_index(drop=True)
    days = pd.read_parquet(ML_DIR / "data" / "dataset.parquet", columns=["video_id", "posted_at"],
                           filters=[("split", "==", "val")])
    va = va.merge(days, on="video_id", how="left", validate="one_to_one")
    y = va["viral"].to_numpy().astype(int)
    PREDS.mkdir(exist_ok=True)
    scores = val_scores(va)
    scores.to_parquet(PREDS / "val_scores.parquet", index=False)

    metrics = json.loads((ML_DIR / "reports" / "metrics.json").read_text())
    recorded = {**metrics["baselines"], **metrics["embeddings"]}
    for k in NAMES:  # recomputed scores must reproduce the recorded metrics exactly
        ap = round(float(average_precision_score(y, scores[k])), 4)
        assert ap == recorded[k]["val"]["pr_auc"], (k, ap, recorded[k]["val"]["pr_auc"])
    log("recomputed validation scores reproduce metrics.json")

    candidates = {k: recorded[k]["val"]["pr_auc"] for k in NAMES}
    if "transformer" in metrics:
        candidates["distilbert"] = metrics["transformer"]["distilbert"]["val"]["pr_auc"]
    best = max(candidates, key=candidates.get)
    blocks = va["posted_at"].dt.strftime("%Y-%m-%d").to_numpy()
    rng = np.random.default_rng(SEED)
    boot = (paired_bootstrap(y, scores[best].to_numpy(), scores["tfidf_lr"].to_numpy(), blocks,
                             rng) if best != "tfidf_lr" else None)
    within_noise = best == "tfidf_lr" or boot["day_block_ci95"][0] <= 0 <= boot["day_block_ci95"][1]
    chosen = "tfidf_lr" if within_noise else best
    log(f"best on val: {best} ({candidates[best]}); bootstrap vs tfidf_lr: {boot}; "
        f"within noise: {within_noise} -> chosen: {chosen}")

    # PR curves on validation
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150, facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    style = {"tfidf_lr": ("#2a78d6", "-"), "tfidf_meta_lgbm": ("#eb6834", "-"),
             "meta_lgbm": ("#1baf7a", "--"), "emb_meta_lgbm": ("#e87ba4", "-.")}
    for k, (color, ls) in style.items():
        p, r, _ = precision_recall_curve(y, scores[k])
        ax.plot(r, p, color=color, linestyle=ls, linewidth=2,
                label=f"{NAMES[k]} (PR-AUC {candidates[k]:.4f})")
    ax.axhline(y.mean(), color="#898781", linewidth=1.2, linestyle=(0, (4, 3)),
               label=f"chance ({y.mean():.4f})")
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Validation precision-recall curves", fontsize=11, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(ML_DIR / "reports" / "val_pr_curves.png", facecolor="#fcfcfb")

    metrics["selection"] = {
        "rule": __doc__.split("Decision rule")[1].split("Also recomputes")[0].strip(),
        "candidates_val_pr_auc": candidates, "best_on_val": best,
        "bootstrap_best_minus_tfidf_lr": boot, "within_noise": bool(within_noise),
        "chosen": chosen, "distilbert_included": "distilbert" in candidates,
    }
    (ML_DIR / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log("saved selection to metrics.json and reports/val_pr_curves.png")


if __name__ == "__main__":
    main()
