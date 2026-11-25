"""Write the README results tables from reports/metrics.json, so they always match exactly.

The tables live between <!-- results:start --> and <!-- results:end --> in README.md.
tests/test_readme_results.py fails if the README and metrics.json disagree.

Usage:  python -m src.readme_results          # rewrite the block
        python -m src.readme_results --check  # exit 1 if the README is out of date
"""

import argparse
import json
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
README = ML_DIR.parent / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"

VALIDATION_ROWS = [  # (section, key, label)
    ("baselines", "majority", "Majority class (constant score)"),
    ("baselines", "random", "Random scores"),
    ("baselines", "meta_lgbm", "Meta features only, LightGBM"),
    ("baselines", "tfidf_lr", "**TF-IDF word + char n-grams, logistic regression (app model)**"),
    ("baselines", "tfidf_meta_lgbm", "TF-IDF + meta features, LightGBM"),
    ("embeddings", "emb_lr", "MiniLM sentence embeddings, logistic regression"),
    ("embeddings", "emb_meta_lgbm", "MiniLM embeddings + meta features, LightGBM"),
    ("transformer", "distilbert", "DistilBERT fine-tuned (~250k-row subset of train)"),
]


def f4(x: float) -> str:
    return f"{x:.4f}"


def render(m: dict) -> str:
    out = ["### Validation (Sep 2025 to Apr 2026, 114,813 posts, 10.01% positive)", "",
           "| Model | PR-AUC | PR-AUC, near-duplicates removed | ROC-AUC | F1 "
           "| Precision in top 10% |",
           "|---|---|---|---|---|---|"]
    for section, key, label in VALIDATION_ROWS:
        r = m.get(section, {}).get(key)
        if r is None:
            continue
        v, d = r["val"], r["val_deduped"]
        out.append(f"| {label} | {f4(v['pr_auc'])} | {f4(d['pr_auc'])} | {f4(v['roc_auc'])} "
                   f"| {f4(v['f1'])} | {f4(v['precision_at_top10'])} |")
    sel = m["selection"]
    b = sel["bootstrap_best_minus_tfidf_lr"]
    out += ["",
            "F1 uses the threshold that maximises F1 on validation "
            f"({f4(m['final']['threshold_from_validation'])} for the app model); the same "
            "threshold is reused unchanged on test.",
            "",
            f"**Model choice.** The best validation PR-AUC was TF-IDF + meta features in LightGBM, "
            f"{b['diff']:+.4f} above TF-IDF + logistic regression. A paired bootstrap over "
            f"{b['n_day_blocks']} posting days gives a 95% interval of "
            f"[{b['day_block_ci95'][0]:+.4f}, {b['day_block_ci95'][1]:+.4f}], which includes 0, "
            "so the simpler model was chosen: it is within noise and gives exact per-word "
            "explanations. DistilBERT was run afterwards as an extra experiment and never "
            "evaluated on test.",
            "",
            "### Test (May to Jul 2026, evaluated once) and stress month (Aug 2026)",
            "",
            "| Split | Posts | Positive rate | PR-AUC | PR-AUC, near-duplicates removed | ROC-AUC "
            "| F1 | Precision in top 10% | Majority PR-AUC | Random PR-AUC |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for split, name in (("test", "Test"), ("stress", "Stress (Aug 2026)")):
        f = m["final"][split]
        r = f["model"]
        out.append(f"| {name} | {r['n']:,} | {f4(r['positive_rate'])} | {f4(r['pr_auc'])} "
                   f"| {f4(f['model_deduped']['pr_auc'])} | {f4(r['roc_auc'])} | {f4(r['f1'])} "
                   f"| {f4(r['precision_at_top10'])} | {f4(f['majority']['pr_auc'])} "
                   f"| {f4(f['random']['pr_auc'])} |")
    sens = m["baselines"]["label_sensitivity"]
    out += ["",
            f"### Label threshold sensitivity (validation, `{sens['model']}`)",
            "",
            "| Label | Positive rate | PR-AUC | PR-AUC / chance | ROC-AUC |",
            "|---|---|---|---|---|"]
    for key, name in (("top_5pct", "Top 5%"), ("top_10pct", "Top 10%"), ("top_20pct", "Top 20%")):
        s = sens[key]
        out.append(f"| {name} | {f4(s['positive_rate'])} | {f4(s['pr_auc'])} "
                   f"| {s['pr_auc_lift_over_chance']:.2f} | {f4(s['roc_auc'])} |")
    shuffled = {**m["baselines"]["sanity_shuffled_labels"],
                **m["embeddings"]["sanity_shuffled_labels"]}
    out += ["",
            "**Leakage checks.** Retraining on shuffled labels gives validation PR-AUC "
            + ", ".join(f4(v) for v in shuffled.values())
            + " (chance is 0.10). Removing validation captions that have a near-duplicate in "
              "train changes PR-AUC by about +0.001."]
    return "\n".join(out)


def block(m: dict) -> str:
    return f"{START}\n{render(m)}\n{END}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    metrics = json.loads((ML_DIR / "reports" / "metrics.json").read_text())
    text = README.read_text(encoding="utf-8")
    i, j = text.find(START), text.find(END)
    if i < 0 or j < 0:
        sys.exit(f"README is missing the {START} / {END} markers")
    current = text[i:j + len(END)]
    wanted = block(metrics)
    if args.check:
        if current != wanted:
            sys.exit("README results are out of date: run python -m src.readme_results")
        print("README results match metrics.json")
        return
    README.write_text(text[:i] + wanted + text[j + len(END):], encoding="utf-8")
    print("README results updated")


if __name__ == "__main__":
    main()
