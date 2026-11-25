"""Phase 3 baselines, evaluated on VALIDATION only (test and stress are not loaded).

Models
  (a) majority (constant score) and random (uniform scores)
  (b) meta features only                -> LightGBM
  (c) TF-IDF word + char n-grams        -> Logistic Regression
  (d) TF-IDF (chi2-selected) + meta     -> LightGBM

Protocol
  - Hyperparameters are chosen by validation PR-AUC over small grids.
  - LightGBM early stopping uses an INNER hold-out = the last 3 training months
    (2025-06..2025-08), not validation, so validation numbers aren't flattered by it.
  - F1 uses the threshold that maximises F1 on validation (saved for use on test later).
  - Validation scores are reported on all rows AND with rows whose caption has a
    near-duplicate in train removed (src/dedup_check.py).
  - Sanity check: the best LR and the meta LightGBM retrained on shuffled training labels.
  - Sensitivity: the best baseline retrained with 5% and 20% labels.

Usage:  python -m src.train_baselines
Writes: reports/metrics.json ("baselines" section), reports/val_per_month_pr_auc.png,
        models/baseline_*.joblib
"""

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate, per_group_pr_auc
from src.features import META_COLUMNS, WORD_TOKEN
from src.labels import label_top_pct

SEED = 42
ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
MODELS = ML_DIR / "models"
INNER_MONTHS = ["2025-06", "2025-07", "2025-08"]



def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load() -> pd.DataFrame:
    cols = ["video_id", "split", "month", "age_bucket", "views", "text", "viral", *META_COLUMNS]
    df = pd.read_parquet(REPO_ROOT / "ml/data/dataset.parquet", columns=cols,
                         filters=[("split", "in", ["train", "val"])])
    flags = pd.read_parquet(REPO_ROOT / "ml/data/dedup_flags.parquet",
                            columns=["video_id", "dup_near_in_train"])
    df = df.merge(flags, on="video_id", how="left", validate="one_to_one")
    assert set(df["split"]) == {"train", "val"}, "test/stress must not be loaded in Phase 3"
    return df.reset_index(drop=True)


def lgb_fit(x_tr, y_tr, x_es, y_es, params: dict) -> lgb.Booster:
    full = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.05,
            "min_child_samples": 100, "feature_fraction": 0.9, "bagging_fraction": 0.8,
            "bagging_freq": 1, "seed": SEED, "verbose": -1, "num_threads": 12, **params}
    return lgb.train(full, lgb.Dataset(x_tr, y_tr), num_boost_round=3000,
                     valid_sets=[lgb.Dataset(x_es, y_es)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def report(name, y_val, s_val, val_df, extra=None) -> dict:
    keep = ~val_df["dup_near_in_train"].to_numpy()
    r = {"val": evaluate(y_val, s_val),
         "val_per_month_pr_auc": per_group_pr_auc(y_val, s_val, val_df["month"]),
         **(extra or {})}
    # deduped F1 uses the threshold tuned on all of val, not re-tuned
    r["val_deduped"] = evaluate(y_val[keep], s_val[keep], threshold=r["val"]["threshold"])
    log(f"{name}: val PR-AUC {r['val']['pr_auc']}, ROC-AUC {r['val']['roc_auc']}, "
        f"P@10% {r['val']['precision_at_top10']}, F1 {r['val']['f1']} | deduped PR-AUC "
        f"{r['val_deduped']['pr_auc']}")
    return r


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    df = load()
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    y_tr, y_va = tr["viral"].to_numpy().astype(int), va["viral"].to_numpy().astype(int)
    inner = tr["month"].isin(INNER_MONTHS).to_numpy()
    log(f"train {len(tr):,} (inner early-stop {inner.sum():,}), val {len(va):,}")
    results: dict = {}

    # ---- (a) trivial baselines ------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    results["majority"] = report("majority", y_va, np.zeros(len(va)), va)
    results["random"] = report("random", y_va, rng.random(len(va)), va)

    # ---- (b) meta-only LightGBM --------------------------------------------------------------
    x_meta_tr = tr[META_COLUMNS].to_numpy(np.float32)
    x_meta_va = va[META_COLUMNS].to_numpy(np.float32)
    best = None
    for leaves in (31, 127):
        bst = lgb_fit(x_meta_tr[~inner], y_tr[~inner], x_meta_tr[inner], y_tr[inner],
                      {"num_leaves": leaves})
        s = bst.predict(x_meta_va)
        r = report(f"meta LGBM leaves={leaves}", y_va, s, va,
                   {"params": {"num_leaves": leaves, "best_iteration": bst.best_iteration},
                    "lgb_params": {"num_leaves": leaves}})
        if best is None or r["val"]["pr_auc"] > best[0]["val"]["pr_auc"]:
            best = (r, bst)
    results["meta_lgbm"], meta_bst = best
    meta_bst.save_model(str(MODELS / "baseline_meta_lgbm.txt"))

    # ---- (c) TF-IDF + LR -----------------------------------------------------------------------
    log("fitting TF-IDF (word 1-2, char_wb 3-5) on train")
    word = TfidfVectorizer(token_pattern=WORD_TOKEN, ngram_range=(1, 2), min_df=3,
                           max_features=300_000, sublinear_tf=True, dtype=np.float32)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           max_features=300_000, sublinear_tf=True, dtype=np.float32)
    x_tr = sp.hstack([word.fit_transform(tr["text"]), char.fit_transform(tr["text"])]).tocsr()
    x_va = sp.hstack([word.transform(va["text"]), char.transform(va["text"])]).tocsr()
    log(f"TF-IDF shape {x_tr.shape}, nnz {x_tr.nnz:,}")
    best = None
    for c in (0.1, 0.3, 1.0):
        t0 = time.time()
        lr = LogisticRegression(C=c, solver="liblinear", max_iter=1000, random_state=SEED)
        lr.fit(x_tr, y_tr)
        r = report(f"TF-IDF LR C={c} ({time.time() - t0:.0f}s)", y_va,
                   lr.predict_proba(x_va)[:, 1], va, {"params": {"C": c}})
        if best is None or r["val"]["pr_auc"] > best[0]["val"]["pr_auc"]:
            best = (r, lr)
    results["tfidf_lr"], lr_best = best
    joblib.dump({"word": word, "char": char, "lr": lr_best}, MODELS / "baseline_tfidf_lr.joblib")

    # ---- (d) TF-IDF (chi2 top-k) + meta -> LightGBM --------------------------------------------
    k = 20_000
    log(f"chi2 selecting top {k:,} TF-IDF features on train")
    sel = SelectKBest(chi2, k=k).fit(x_tr, y_tr)
    xd_tr = sp.hstack([sel.transform(x_tr), sp.csr_matrix(x_meta_tr)]).tocsr()
    xd_va = sp.hstack([sel.transform(x_va), sp.csr_matrix(x_meta_va)]).tocsr()
    best = None
    for leaves in (63, 255):
        t0 = time.time()
        params = {"num_leaves": leaves, "learning_rate": 0.1, "feature_fraction": 0.3}
        bst = lgb_fit(xd_tr[~inner], y_tr[~inner], xd_tr[inner], y_tr[inner], params)
        r = report(f"TF-IDF+meta LGBM leaves={leaves} ({time.time() - t0:.0f}s)", y_va,
                   bst.predict(xd_va), va,
                   {"params": {"num_leaves": leaves, "k_tfidf": k,
                               "best_iteration": bst.best_iteration},
                    "lgb_params": params})
        if best is None or r["val"]["pr_auc"] > best[0]["val"]["pr_auc"]:
            best = (r, bst)
    results["tfidf_meta_lgbm"], d_bst = best
    d_bst.save_model(str(MODELS / "baseline_tfidf_meta_lgbm.txt"))
    joblib.dump(sel, MODELS / "baseline_tfidf_meta_selector.joblib")

    # ---- sanity: shuffled training labels -----------------------------------------------------
    y_shuf = rng.permutation(y_tr)
    lr_s = LogisticRegression(C=results["tfidf_lr"]["params"]["C"], solver="liblinear",
                              max_iter=1000, random_state=SEED).fit(x_tr, y_shuf)
    meta_s = lgb_fit(x_meta_tr[~inner], y_shuf[~inner], x_meta_tr[inner], y_shuf[inner],
                     {"num_leaves": results["meta_lgbm"]["params"]["num_leaves"]})
    results["sanity_shuffled_labels"] = {
        "tfidf_lr_val_pr_auc": evaluate(y_va, lr_s.predict_proba(x_va)[:, 1])["pr_auc"],
        "meta_lgbm_val_pr_auc": evaluate(y_va, meta_s.predict(x_meta_va))["pr_auc"],
    }
    log(f"shuffled-label sanity: {results['sanity_shuffled_labels']}")

    # ---- sensitivity: best baseline with 5% / 20% labels ---------------------------------------
    candidates = ["meta_lgbm", "tfidf_lr", "tfidf_meta_lgbm"]
    best_name = max(candidates, key=lambda n: results[n]["val"]["pr_auc"])
    log(f"best baseline on val PR-AUC: {best_name}")
    sens = {"model": best_name}
    for pct in (0.05, 0.10, 0.20):
        y_all = label_top_pct(df, pct, ["month", "age_bucket"]).to_numpy().astype(int)
        yt, yv = y_all[df["split"].to_numpy() == "train"], y_all[df["split"].to_numpy() == "val"]
        if best_name == "tfidf_lr":
            m = LogisticRegression(C=results["tfidf_lr"]["params"]["C"], solver="liblinear",
                                   max_iter=1000, random_state=SEED).fit(x_tr, yt)
            s = m.predict_proba(x_va)[:, 1]
        else:
            xt, xv = (x_meta_tr, x_meta_va) if best_name == "meta_lgbm" else (xd_tr, xd_va)
            m = lgb_fit(xt[~inner], yt[~inner], xt[inner], yt[inner],
                        results[best_name]["lgb_params"])
            s = m.predict(xv)
        e = evaluate(yv, s)
        e["pr_auc_lift_over_chance"] = round(e["pr_auc"] / e["positive_rate"], 2)
        sens[f"top_{int(pct * 100)}pct"] = e
        log(f"sensitivity {pct:.0%}: {e}")
    results["label_sensitivity"] = sens
    results["best_baseline"] = best_name

    # ---- save -----------------------------------------------------------------------------------
    path = ML_DIR / "reports" / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics["baselines"] = {"protocol": __doc__.split("Protocol")[1].split("Usage")[0].strip(),
                            "evaluated_on": "validation 2025-09..2026-04 (test untouched)",
                            **results}
    path.write_text(json.dumps(metrics, indent=2))
    log(f"saved {path}")


if __name__ == "__main__":
    main()
