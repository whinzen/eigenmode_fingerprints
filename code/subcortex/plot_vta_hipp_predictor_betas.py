#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

HC_GROUP = (
    PANG
    / "subcortex"
    / "hipp_mean_signal_group"
    / "group_all_predictors_hipp_mean_signal_bihemi.csv"
)

BS_GROUP = (
    PANG
    / "subcortex"
    / "brainstem_roi_group"
    / "group_brainstem_roi_glm_all_predictors.csv"
)

OUT = PANG / "subcortex" / "brainstem_vta_hipp_figures"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = [
    "sentence_boundary",
    "sentence_shift",
    "token_shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]

LABELS = {
    "sentence_boundary": "Sentence\nboundary",
    "sentence_shift": "Sentence\nshift",
    "token_shift": "Token\nshift",
    "pred_error_ar": "AR\nerror",
    "pred_error_subspace": "Subspace\nerror",
    "curvature": "Curvature",
}

SCATTER_LABELS = {
    "sentence_boundary": "Sentence boundary",
    "sentence_shift": "Sentence shift",
    "token_shift": "Token shift",
    "pred_error_ar": "AR error",
    "pred_error_subspace": "Subspace error",
    "curvature": "Curvature",
}

# Consistent predictor colors across panels.
# These are matplotlib default/tab colors but fixed explicitly for reproducibility.
COLORS = {
    "sentence_boundary": "#4C72B0",
    "sentence_shift": "#55A868",
    "token_shift": "#C44E52",
    "pred_error_ar": "#8172B2",
    "pred_error_subspace": "#CCB974",
    "curvature": "#64B5CD",
}

# Manual label positions for Panel C.
# Values are absolute offsets in beta-coordinate space.
OFFSETS = {
    "sentence_boundary": (-0.020, -0.011),
    "sentence_shift": (0.017, 0.010),
    "token_shift": (-0.018, 0.011),
    "pred_error_ar": (0.015, 0.006),
    "pred_error_subspace": (0.020, -0.014),
    "curvature": (-0.018, -0.018),
}


def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_stars(ax, x, y, err, pvals):
    ymin, ymax = ax.get_ylim()
    offset = 0.04 * (ymax - ymin)

    for xi, yi, ei, p in zip(x, y, err, pvals):
        s = stars(p)

        if yi < 0:
            ypos = yi - ei - offset
            va = "top"
        else:
            ypos = yi + ei + offset
            va = "bottom"

        ax.text(
            xi,
            ypos,
            s,
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )


def prepare_data():
    hc = pd.read_csv(HC_GROUP)
    bs = pd.read_csv(BS_GROUP)

    # VTA only; average left/right descriptively at group level.
    vta = bs[bs["roi"].isin(["VTA_L", "VTA_R"])].copy()

    vta = (
        vta.groupby("predictor", as_index=False)
        .agg(
            beta_mean=("beta_mean", "mean"),
            beta_sem=("beta_sem", "mean"),
            p=("p", "max"),
            q=("q", "max"),
        )
    )

    hc = hc[hc["predictor"].isin(ORDER)].copy()
    vta = vta[vta["predictor"].isin(ORDER)].copy()

    hc["predictor"] = pd.Categorical(
        hc["predictor"],
        ORDER,
        ordered=True,
    )

    vta["predictor"] = pd.Categorical(
        vta["predictor"],
        ORDER,
        ordered=True,
    )

    hc = hc.sort_values("predictor")
    vta = vta.sort_values("predictor")

    merged = hc[["predictor", "beta_mean"]].merge(
        vta[["predictor", "beta_mean"]],
        on="predictor",
        suffixes=("_HC", "_VTA"),
    )

    return hc, vta, merged


def main():
    hc, vta, merged = prepare_data()

    r, p_corr = stats.pearsonr(
        merged["beta_mean_VTA"],
        merged["beta_mean_HC"],
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.2, 5.3),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.30]},
    )

    # ------------------------------------------------------------------
    # Panel A: hippocampus
    # ------------------------------------------------------------------
    ax = axes[0]

    x = np.arange(len(hc)) * 1.20
    predictors = [str(p) for p in hc["predictor"]]
    colors = [COLORS[p] for p in predictors]

    y = hc["beta_mean"].values
    err = hc["beta_sem"].values
    pvals = hc["p"].values

    ax.bar(
        x,
        y,
        yerr=err,
        capsize=4,
        width=0.60,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.axhline(0, linewidth=1, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS[p] for p in predictors],
        fontsize=9,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("β", fontsize=13)
    ax.set_title(
        "A. Hippocampal mean signal",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylim(-0.205, 0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_stars(ax, x, y, err, pvals)

    # ------------------------------------------------------------------
    # Panel B: VTA
    # ------------------------------------------------------------------
    ax = axes[1]

    x = np.arange(len(vta)) * 1.20
    predictors = [str(p) for p in vta["predictor"]]
    colors = [COLORS[p] for p in predictors]

    y = vta["beta_mean"].values
    err = vta["beta_sem"].values
    pvals = vta["p"].values

    ax.bar(
        x,
        y,
        yerr=err,
        capsize=4,
        width=0.60,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.axhline(0, linewidth=1, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS[p] for p in predictors],
        fontsize=9,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("β", fontsize=13)
    ax.set_title(
        "B. VTA mean signal",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylim(-0.205, 0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_stars(ax, x, y, err, pvals)

    # ------------------------------------------------------------------
    # Panel C: profile similarity
    # ------------------------------------------------------------------
    ax = axes[2]

    for _, row in merged.iterrows():
        pred = str(row["predictor"])

        ax.scatter(
            row["beta_mean_VTA"],
            row["beta_mean_HC"],
            s=80,
            color=COLORS[pred],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    xvals = merged["beta_mean_VTA"].values
    yvals = merged["beta_mean_HC"].values

    slope, intercept, _, _, _ = stats.linregress(xvals, yvals)
    xx = np.linspace(xvals.min() - 0.008, xvals.max() + 0.008, 100)

    ax.plot(
        xx,
        intercept + slope * xx,
        linewidth=1.7,
        color="black",
        zorder=2,
    )

    for _, row in merged.iterrows():
        pred = str(row["predictor"])
        dx, dy = OFFSETS[pred]

        ax.annotate(
            SCATTER_LABELS[pred],
            xy=(row["beta_mean_VTA"], row["beta_mean_HC"]),
            xytext=(
                row["beta_mean_VTA"] + dx,
                row["beta_mean_HC"] + dy,
            ),
            fontsize=8.8,
            ha="center",
            va="center",
            color=COLORS[pred],
            arrowprops=dict(
                arrowstyle="-",
                color=COLORS[pred],
                lw=0.8,
                shrinkA=0,
                shrinkB=4,
            ),
        )

    ax.axhline(0, linewidth=1, color="black")
    ax.axvline(0, linewidth=1, color="black")

    ax.set_xlabel("VTA β", fontsize=13)
    ax.set_ylabel("Hippocampus β", fontsize=13)

    ax.set_title(
        f"C. Predictor-profile similarity\nr = {r:.4f}, p = {p_corr:.2e}",
        fontsize=13,
        fontweight="bold",
    )

    # Expanded limits to create room for labels.
    ax.set_xlim(-0.172, -0.025)
    ax.set_ylim(-0.175, -0.030)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.2)

    out_png = OUT / "vta_hipp_predictor_betas.png"
    out_pdf = OUT / "vta_hipp_predictor_betas.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    caption = """Figure X. Parallel hippocampal and VTA responses to linguistic transition predictors.
(A) Group-level GLM coefficients for mean hippocampal BOLD signal.
(B) Group-level GLM coefficients for VTA mean BOLD signal, averaged descriptively across left and right VTA masks.
(C) Across-predictor similarity between hippocampal and VTA response profiles. Each point corresponds to one linguistic predictor, with colors matched across panels. The close alignment between hippocampal and VTA beta profiles suggests that both regions are sensitive to a common transition-related signal, with weaker effects for sentence-level predictors and stronger effects for token-level representational transition metrics. Error bars indicate SEM across subjects after run-level averaging. Stars denote one-sample tests against zero: * p < .05, ** p < .01, *** p < .001, **** p < 1e-4.
"""

    caption_file = OUT / "vta_hipp_predictor_betas_caption.txt"
    caption_file.write_text(caption)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {caption_file}")

    print("\nMerged profile:")
    print(merged)

    print(f"\nProfile correlation: r={r:.4f}, p={p_corr:.4g}")


if __name__ == "__main__":
    main()