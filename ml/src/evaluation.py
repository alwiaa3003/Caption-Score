"""Metrics shared by every model. PR-AUC is primary (chance = positive rate, ~0.10)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold that maximises F1 on (y, scores). Returns (threshold, f1).

    Tune this on validation only; on test, apply the validation threshold unchanged.
    """
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0  # last point has no threshold
    return float(thresholds[i]) if len(thresholds) else 0.5, float(f1[i])


def f1_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    pred = scores >= threshold
    tp = float((pred & (y == 1)).sum())
    denom = pred.sum() + (y == 1).sum()
    return 2 * tp / denom if denom else 0.0


def precision_at_top(y: np.ndarray, scores: np.ndarray, frac: float = 0.10) -> float:
    """Precision among the `frac` highest-scored rows (ties broken by a stable sort)."""
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-scores, kind="stable")[:k]
    return float(y[top].mean())


def evaluate(y, scores, threshold: float | None = None) -> dict:
    """All metrics. If `threshold` is None it is tuned here (only legitimate on validation)."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    out = {
        "n": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "precision_at_top10": round(precision_at_top(y, scores, 0.10), 4),
    }
    if threshold is None:
        threshold, f1 = best_f1_threshold(y, scores)
    else:
        f1 = f1_at(y, scores, threshold)
    out["f1"] = round(f1, 4)
    out["threshold"] = round(float(threshold), 6)
    return out


def per_group_pr_auc(y, scores, groups) -> dict[str, float]:
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "s": scores, "g": np.asarray(groups)})
    return {g: round(float(average_precision_score(d["y"], d["s"])), 4)
            for g, d in df.groupby("g", sort=True)}
