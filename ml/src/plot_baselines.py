"""Plot per-month validation PR-AUC for each baseline, from reports/metrics.json only.

Usage:  python -m src.plot_baselines
Writes: reports/val_per_month_pr_auc.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ML_DIR = Path(__file__).resolve().parents[1]

# Reference categorical palette, slots 1-3 in fixed order (validated: CVD + normal vision pass;
# aqua is < 3:1 on the surface, so every line is also direct-labelled and has its own marker).
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
SERIES = [
    ("tfidf_lr", "TF-IDF + LR (text only)", "#2a78d6", "o"),
    ("tfidf_meta_lgbm", "TF-IDF + meta (LightGBM)", "#eb6834", "s"),
    ("meta_lgbm", "Meta only (LightGBM)", "#1baf7a", "^"),
]


def main() -> None:
    b = json.loads((ML_DIR / "reports" / "metrics.json").read_text())["baselines"]
    months = list(b["tfidf_lr"]["val_per_month_pr_auc"].keys())
    x = range(len(months))

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for key, label, color, marker in SERIES:
        ys = [b[key]["val_per_month_pr_auc"][m] for m in months]
        ax.plot(x, ys, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.annotate(f"{label}  {b[key]['val']['pr_auc']:.4f} overall", (len(months) - 1, ys[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK)
    chance = b["random"]["val"]["positive_rate"]
    ax.axhline(chance, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"chance = positive rate ({chance:.2f})", (0, chance), xytext=(0, -12),
                textcoords="offset points", fontsize=8, color=MUTED)

    ax.set_xticks(list(x), months, fontsize=8, color=INK)
    ax.tick_params(axis="y", labelsize=8, colors=INK)
    ax.tick_params(length=0)
    ax.set_ylim(0, None)
    ax.set_ylabel("PR-AUC", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_title("Validation PR-AUC by posting month (US, English, top-10% label)",
                 fontsize=11, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncols=3,
              bbox_to_anchor=(0, -0.1))
    fig.subplots_adjust(right=0.72, bottom=0.2)
    out = ML_DIR / "reports" / "val_per_month_pr_auc.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
