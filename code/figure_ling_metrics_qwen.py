#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN_CSV = PANG / "word_transition_geometry_qwen3_0p6b" / "word_transition_geometry.csv"
OUT_FIG = PANG / "paper_figures"
OUT_TAB = PANG / "paper_tables"

OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

METRICS = {
    "shift": "Token shift",
    "pred_error_ar": "Prediction error",
    "pred_error_subspace": "Subspace exit",
    "curvature": "Curvature",
}


# =========================
# Helpers
# =========================
def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.nan if len(x) < 2 else x.std(ddof=1) / np.sqrt(len(x))


def cohens_d(boundary, within):
    boundary = np.asarray(boundary, float)
    within = np.asarray(within, float)

    boundary = boundary[np.isfinite(boundary)]
    within = within[np.isfinite(within)]

    n1, n0 = len(boundary), len(within)
    if n1 < 2 or n0 < 2:
        return np.nan

    s1 = boundary.std(ddof=1)
    s0 = within.std(ddof=1)

    pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    if pooled == 0:
        return np.nan

    return (boundary.mean() - within.mean()) / pooled


def stars(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(IN_CSV)

    if "same_run" in df.columns:
        df = df[df["same_run"] == 1].copy()

    rows = []

    for metric, label in METRICS.items():
        within = df.loc[df["boundary_flag"] == 0, metric].dropna().values
        boundary = df.loc[df["boundary_flag"] == 1, metric].dropna().values

        t, p = ttest_ind(boundary, within, equal_var=False, nan_policy="omit")
        d = cohens_d(boundary, within)

        rows.append({
            "metric": metric,
            "label": label,
            "within_mean": within.mean(),
            "within_sem": sem(within),
            "boundary_mean": boundary.mean(),
            "boundary_sem": sem(boundary),
            "p": p,
            "d": d,
            "stars": stars(p),
        })

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_TAB / "table_ling_metrics_qwen.csv", index=False)

    # =========================
    # Plot
    # =========================
    x = np.arange(len(tab))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.bar(
        x - width / 2,
        tab["within_mean"],
        width,
        yerr=tab["within_sem"],
        capsize=4,
        label="Within sentence",
    )

    ax.bar(
        x + width / 2,
        tab["boundary_mean"],
        width,
        yerr=tab["boundary_sem"],
        capsize=4,
        label="Sentence boundary",
    )

    # =========================
    # Significance annotations
    # =========================
    y_max = np.maximum(
        tab["within_mean"] + tab["within_sem"],
        tab["boundary_mean"] + tab["boundary_sem"],
    )

    y_range = y_max.max() - y_max.min()

    for i, row in tab.iterrows():
        y = y_max.iloc[i] + 0.08 * y_range
        h = 0.02 * y_range

        x1 = x[i] - width / 2
        x2 = x[i] + width / 2

        # bracket
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1)

        # stars + effect size
        ax.text(
            x[i],
            y + h + 0.01 * y_range,
            f"{row['stars']}\nd={row['d']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tab["label"], rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Linguistic transition metrics show boundary effects (Qwen3)")

    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(top=y_max.max() + 0.25 * y_range)

    plt.tight_layout()

    plt.savefig(OUT_FIG / "figure_ling_metrics_qwen.png", dpi=300)
    plt.savefig(OUT_FIG / "figure_ling_metrics_qwen.pdf")

    print("✅ figure + table written")
    print(tab)


if __name__ == "__main__":
    main()