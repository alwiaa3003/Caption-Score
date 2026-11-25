import json

from src.readme_results import END, ML_DIR, README, START, block


def test_readme_results_match_metrics_json():
    metrics = json.loads((ML_DIR / "reports" / "metrics.json").read_text())
    text = README.read_text(encoding="utf-8")
    current = text[text.index(START):text.index(END) + len(END)]
    assert current == block(metrics), "run: python -m src.readme_results"


def test_app_model_numbers_appear_in_readme():
    metrics = json.loads((ML_DIR / "reports" / "metrics.json").read_text())
    text = README.read_text(encoding="utf-8")
    for value in (metrics["baselines"]["tfidf_lr"]["val"]["pr_auc"],
                  metrics["final"]["test"]["model"]["pr_auc"],
                  metrics["final"]["stress"]["model"]["pr_auc"]):
        assert f"{value:.4f}" in text
