"""Phase 5: global explanations, hashtag lift table, calibration check, and the final model
bundle used by the model service.

Everything is computed on TRAIN (lift, n-gram frequencies, hashtag centroids) or VALIDATION
(calibration, label cut-off). Test and stress are not read here.
Only aggregates are written (n-grams/hashtags seen in >= 500 training posts), never captions.

Usage:  python -m src.build_explainability
Writes: reports/explainability.md, reports/explainability.json, reports/hashtag_lift.csv,
        models/final/{model.joblib, suggest.joblib, model_card.json}
"""

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.explain import CaptionModel, HashtagSuggester, display_piece
from src.train_baselines import MODELS, log

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
FINAL = MODELS / "final"
MIN_COUNT = 500
TOP_K = 25
CENTROID_TERMS = 300
MODEL_VERSION = "tfidf-lr-v1"

# Hand-written examples for the report (never dataset captions).
EXAMPLES = [
    "wait for the end 😂😂 #funny #comedy",
    "my 5 minute morning routine #grwm #morningroutine",
    "link in bio for the full recipe #food #recipe",
    "storytime: the worst date of my life (part 1)",
]


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def hashtag_lift(train: pd.DataFrame, base: float) -> pd.DataFrame:
    e = train[["hashtags", "viral"]].explode("hashtags").dropna()
    g = e.groupby("hashtags")["viral"].agg(posts="size", viral_posts="sum")
    g = g[g["posts"] >= MIN_COUNT].copy()
    g["viral_rate"] = g["viral_posts"] / g["posts"]
    g["lift"] = g["viral_rate"] / base
    lo, hi = wilson(g["viral_posts"].to_numpy(float), g["posts"].to_numpy(float))
    g["rate_ci95_low"], g["rate_ci95_high"] = lo, hi
    return g.sort_values("lift", ascending=False).round(4).rename_axis("hashtag").reset_index()


def main() -> None:
    t0 = time.time()
    bundle = joblib.load(MODELS / "baseline_tfidf_lr.joblib")
    model = CaptionModel(bundle)
    df = pd.read_parquet(REPO_ROOT / "ml/data/dataset.parquet",
                         columns=["split", "text", "hashtags", "viral"],
                         filters=[("split", "in", ["train", "val"])])
    assert set(df["split"]) == {"train", "val"}
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    base = float(train["viral"].mean())
    log(f"train {len(train):,}, val {len(val):,}, base rate {base:.4f}")

    # ---- 1. global word / hashtag features ---------------------------------------------------
    xw = model.word.transform(train["text"])
    doc_freq = np.asarray((xw > 0).sum(axis=0)).ravel()
    y = train["viral"].to_numpy().astype(float)
    viral_with = np.asarray((xw > 0).T @ y).ravel()
    feats = pd.DataFrame({"feature": model.word.get_feature_names_out(), "coef": model.w_word,
                          "train_posts": doc_freq})
    feats["viral_rate_when_present"] = np.where(doc_freq > 0, viral_with / np.maximum(doc_freq, 1),
                                                np.nan)
    feats = feats[feats["train_posts"] >= MIN_COUNT]
    is_tag = feats["feature"].str.startswith("#") & ~feats["feature"].str.contains(" ")
    words, tags = feats[~is_tag], feats[is_tag]

    def top(frame, ascending):
        out = frame.sort_values("coef", ascending=ascending).head(TOP_K).copy()
        out["feature"] = out["feature"].map(lambda f: " ".join(display_piece(p)
                                                               for p in f.split(" ")))
        return out.round(4).to_dict(orient="records")

    global_tables = {"ngrams_positive": top(words, False), "ngrams_negative": top(words, True),
                     "hashtags_positive": top(tags, False), "hashtags_negative": top(tags, True)}
    log("global feature tables done")

    # ---- 2. hashtag lift table ----------------------------------------------------------------
    lift = hashtag_lift(train, base)
    lift.to_csv(ML_DIR / "reports" / "hashtag_lift.csv", index=False)
    cand = lift[lift["rate_ci95_low"] > base].reset_index(drop=True)
    log(f"hashtag lift: {len(lift)} tags with >= {MIN_COUNT} posts; "
        f"{len(cand)} with CI lower bound above base rate (suggestion pool)")

    # centroids: mean word-TF-IDF of training captions containing each candidate tag
    tag_index = {t: i for i, t in enumerate(cand["hashtag"])}
    rows, cols = [], []
    for r, tags_ in enumerate(train["hashtags"]):
        for t in tags_:
            if t in tag_index:
                rows.append(tag_index[t])
                cols.append(r)
    member = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cand), len(train)))
    member = normalize(member, norm="l1")
    centroids = (member @ xw).tocsr()
    pruned = []
    for i in range(centroids.shape[0]):  # keep each centroid's strongest terms only
        row = centroids.getrow(i)
        keep = np.argsort(-row.data)[:CENTROID_TERMS]
        pruned.append(sp.csr_matrix((row.data[keep], (np.zeros(len(keep)), row.indices[keep])),
                                    shape=(1, centroids.shape[1])))
    centroids = normalize(sp.vstack(pruned).tocsr())
    suggest_bundle = {"tags": cand["hashtag"].tolist(), "lift": cand["lift"].to_numpy(),
                      "centroids": centroids}

    # ---- 3. calibration + label cut-off on validation ------------------------------------------
    p_val = model.predict_proba(val["text"].tolist())
    yv = val["viral"].to_numpy().astype(int)
    cut = float(np.quantile(p_val, 0.9))  # "likely top 10%" = model's top-decile score
    dec = pd.qcut(p_val, 10, labels=False, duplicates="drop")
    calib = (pd.DataFrame({"decile": dec, "p": p_val, "y": yv}).groupby("decile")
             .agg(mean_predicted=("p", "mean"), actual_rate=("y", "mean"), rows=("y", "size"))
             .round(4).reset_index().to_dict(orient="records"))
    above = p_val >= cut
    label_info = {"likely_threshold": round(cut, 6),
                  "definition": "score in the top 10% of validation scores",
                  "val_share_labelled_likely": round(float(above.mean()), 4),
                  "val_precision_when_likely": round(float(yv[above].mean()), 4),
                  "val_viral_rate_when_unlikely": round(float(yv[~above].mean()), 4)}
    log(f"label cut-off {label_info}")

    # ---- 4. hand-written examples -------------------------------------------------------------
    suggester = HashtagSuggester(suggest_bundle, model)
    examples = []
    for text in EXAMPLES:
        exp = model.explain(text)
        pos, neg = model.top_tokens(exp, k=4)
        examples.append({"caption": text, "probability": round(exp.probability, 4),
                         "likely": bool(exp.probability >= cut), "top_positive": pos,
                         "top_negative": neg, "suggested_hashtags": suggester.suggest(text, k=3)})

    # ---- 5. export final bundle --------------------------------------------------------------
    if FINAL.exists():
        shutil.rmtree(FINAL)
    FINAL.mkdir(parents=True)
    joblib.dump(bundle, FINAL / "model.joblib", compress=3)
    joblib.dump(suggest_bundle, FINAL / "suggest.joblib", compress=3)
    metrics = json.loads((ML_DIR / "reports" / "metrics.json").read_text())
    card = {
        "model_version": MODEL_VERSION,
        "model": "TF-IDF (word 1-2 + char_wb 3-5 n-grams) + Logistic Regression",
        "trained_on": "TikTok, US, English videos posted 2022-01..2025-08 (656,136 posts)",
        "label": "top 10% of views among posts from the same month and age-at-capture tertile",
        "base_rate": round(base, 4),
        "label_cutoff": label_info,
        "validation": metrics["baselines"]["tfidf_lr"]["val"],
        "test": metrics["final"]["test"]["model"],
        "stress_2026_08": metrics["final"]["stress"]["model"],
        "caveats": ["trained on TikTok data; Instagram use is an experiment",
                    "creator size is not observed, so captions explain only a small part",
                    "hashtag suggestions are associations, not causal effects"],
    }
    (FINAL / "model_card.json").write_text(json.dumps(card, indent=2))
    sizes = {f.name: round(f.stat().st_size / 1e6, 1) for f in FINAL.iterdir()}
    log(f"exported {FINAL} {sizes} MB")

    report = {"min_count": MIN_COUNT, "base_rate": round(base, 4), **global_tables,
              "hashtags_with_min_count": len(lift), "suggestion_pool": len(cand),
              "calibration_val_deciles": calib, "label_cutoff": label_info,
              "examples": examples, "bundle_sizes_mb": sizes}
    (ML_DIR / "reports" / "explainability.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, lift)
    log(f"done in {time.time() - t0:.0f}s")


def write_markdown(r: dict, lift: pd.DataFrame) -> None:
    def tbl(rows, cols):
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        return head + "".join("| " + " | ".join(str(x[c]) for c in cols) + " |\n" for x in rows)

    c = ["feature", "coef", "train_posts", "viral_rate_when_present"]
    lab = r["label_cutoff"]
    lc = ["hashtag", "posts", "viral_rate", "lift", "rate_ci95_low", "rate_ci95_high"]
    md = [
        "# Explainability (final model: TF-IDF + LR)\n",
        "Generated by `python -m src.build_explainability`. Features shown were seen in at least "
        f"{r['min_count']} training posts. `coef` is the logistic-regression weight on the TF-IDF "
        "value (positive pushes towards top-10%). `viral_rate_when_present` is descriptive: "
        f"base rate {r['base_rate']}. Char n-gram weights also contribute to every prediction "
        "but aren't listed here.\n",
        "## Words / phrases that push the score up\n", tbl(r["ngrams_positive"], c),
        "\n## Words / phrases that push the score down\n", tbl(r["ngrams_negative"], c),
        "\n## Hashtag features: up\n", tbl(r["hashtags_positive"], c),
        "\n## Hashtag features: down\n", tbl(r["hashtags_negative"], c),
        f"\n## Hashtag lift (training set, >= {r['min_count']} posts)\n",
        f"{r['hashtags_with_min_count']} hashtags qualify; {r['suggestion_pool']} have a 95% "
        "Wilson lower bound above the base rate and form the suggestion pool. "
        "Full table: `hashtag_lift.csv`.\n\n**Highest lift**\n\n",
        tbl(lift.head(20).to_dict(orient="records"), lc),
        "\n**Lowest lift**\n\n", tbl(lift.tail(10).iloc[::-1].to_dict(orient="records"), lc),
        "\n## Calibration on validation (deciles of predicted probability)\n",
        tbl(r["calibration_val_deciles"], ["decile", "mean_predicted", "actual_rate", "rows"]),
        f"\n## Label used by the app\n`likely` if probability >= {lab['likely_threshold']}"
        " (the model's top-10% score on validation). On validation, "
        f"{lab['val_share_labelled_likely']:.1%} of posts were `likely`, and "
        f"{lab['val_precision_when_likely']:.1%} of those were actually top-10% "
        f"(vs {lab['val_viral_rate_when_unlikely']:.1%} for `unlikely`).\n",
        "\n## Example explanations (hand-written captions)\n",
    ]
    for e in r["examples"]:
        verdict = "likely" if e["likely"] else "unlikely"
        up = ", ".join(t["token"] for t in e["top_positive"])
        down = ", ".join(t["token"] for t in e["top_negative"])
        sugg = ", ".join("#" + s["hashtag"] for s in e["suggested_hashtags"]) or "none"
        md.append(f"- `{e['caption']}` -> p={e['probability']} ({verdict}); up: {up}; "
                  f"down: {down}; suggest: {sugg}\n")
    (ML_DIR / "reports" / "explainability.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
