import numpy as np

from src.evaluation import best_f1_threshold, evaluate, f1_at, precision_at_top


def test_perfect_scores():
    y = np.array([0] * 90 + [1] * 10)
    s = y.astype(float)
    e = evaluate(y, s)
    assert e["pr_auc"] == 1.0 and e["roc_auc"] == 1.0
    assert e["precision_at_top10"] == 1.0 and e["f1"] == 1.0


def test_random_scores_are_near_chance():
    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.1).astype(int)
    e = evaluate(y, rng.random(50_000))
    assert abs(e["pr_auc"] - 0.1) < 0.01
    assert abs(e["roc_auc"] - 0.5) < 0.01


def test_precision_at_top_counts_the_top_fraction():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    assert precision_at_top(y, s, 0.2) == 0.5
    assert precision_at_top(y, s, 0.1) == 1.0


def test_tuned_threshold_reproduces_f1():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.2).astype(int)
    s = y * 0.3 + rng.random(2000)
    t, f1 = best_f1_threshold(y, s)
    assert abs(f1_at(y, s, t) - f1) < 1e-9
