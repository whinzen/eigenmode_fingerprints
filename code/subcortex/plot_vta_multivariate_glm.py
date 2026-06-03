#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN = (
    PANG
    / "subcortex"
    / "vta_multivariate_glm"
    / "vta_multivariate_glm_bilateral_group.csv"
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

COLORS = {
    "sentence_boundary": "#4C72B0",
    "sentence_shift": "#55A868",
    "token_shift": "#C44E52",
    "pred_error_ar": "#8172B2",
    "pred_error_subspace": "#CCB974",
    "curvature": "#64B5CD",
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


def main():
    df = pd.read_csv(IN)
    df = df[df["predictor"].isin(ORDER)].copy()
    df["predictor"] = pd.Categorical(df["predictor"], ORDER, ordered=True)
    df = df.sort_values("predictor")

    x = np.arange(len(df)) * 0.9
    y = df["beta_mean"].values
    err = df["beta_sem"].values
    pvals = df["p"].values
    preds = [str(p) for p in df["predictor"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.bar(
        x,
        y,
        yerr=err,
        capsize=4,
        width=0.58,
        color=[COLORS[p] for p in preds],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[p] for p in preds], fontsize=9)
    ax.set_ylabel("Unique β", fontsize=12)
    ax.set_title(
        "Multivariate VTA GLM: unique predictor effects",
        fontsize=12,
        fontweight="bold",
    )

    ymin = min(y - err) - 0.035
    ymax = max(y + err) + 0.035
    ax.set_ylim(ymin, ymax)

    for xi, yi, ei, p in zip(x, y, err, pvals):
        ypos = yi - ei - 0.015 if yi < 0 else yi + ei + 0.015
        va = "top" if yi < 0 else "bottom"
        ax.text(
            xi,
            ypos,
            stars(p),
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_png = OUT / "vta_multivariate_glm_unique_predictors.png"
    out_pdf = OUT / "vta_multivariate_glm_unique_predictors.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    caption = """Figure X. Unique VTA effects in a multivariate GLM. Bilateral VTA mean signal was modeled using all six linguistic and representational predictors simultaneously: sentence boundary, sentence-level representational shift, token shift, autoregressive prediction error, subspace prediction error, and curvature. Bars show group-level mean β estimates after averaging runwise coefficients within subject and averaging left and right VTA descriptively. Error bars indicate SEM across subjects. Sentence-boundary and sentence-shift effects remained significant after controlling for continuous token-level transition metrics, indicating unique boundary-related VTA variance. However, representational curvature emerged as the strongest independent predictor, suggesting that VTA activity is particularly sensitive to changes in the direction of the evolving representational trajectory. Stars denote one-sample tests against zero: * p < .05, ** p < .01, *** p < .001, **** p < 1e-4."""
    caption_file = OUT / "vta_multivariate_glm_unique_predictors_caption.txt"
    caption_file.write_text(caption)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {caption_file}")
    print(df)


if __name__ == "__main__":
    main()