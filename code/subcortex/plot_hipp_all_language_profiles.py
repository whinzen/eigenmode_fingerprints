#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN_DIR = PANG / "subcortex" / "hippocampus_group"
OUT_DIR = PANG / "subcortex" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Sentence boundary": IN_DIR / "group_boundary_hipp_bihemi_by_mode_subject_level.csv",
    "Sentence shift": IN_DIR / "group_sentence_shift_hipp_bihemi_by_mode_subject_level.csv",
    "Token shift": IN_DIR / "group_token_shift_hipp_bihemi_by_mode_subject_level.csv",
    "Prediction error": IN_DIR / "group_pred_error_ar_hipp_bihemi_by_mode_subject_level.csv",
    "Subspace exit": IN_DIR / "group_pred_error_subspace_hipp_bihemi_by_mode_subject_level.csv",
    "Curvature": IN_DIR / "group_curvature_hipp_bihemi_by_mode_subject_level.csv",
}

COLORS = {
    "Sentence boundary": "#1f77b4",
    "Sentence shift": "#d62728",
    "Token shift": "#2ca02c",
    "Prediction error": "#9467bd",
    "Subspace exit": "#ff7f0e",
    "Curvature": "#8c564b",
}


def load(path):
    df = pd.read_csv(path)
    df = df[df["mode_k"] > 0].sort_values("mode_k").copy()
    return df


def plot_metric(ax, label, path):
    df = load(path)
    c = COLORS[label]

    x = df["mode_k"]
    y = df["beta_mean"]
    sem = df["beta_sem"]

    ax.plot(x, y, linewidth=2.2, label=label, color=c)
    ax.fill_between(x, y - sem, y + sem, alpha=0.14, color=c)

    if "significant_fdr05_bi" in df.columns:
        sig = df["significant_fdr05_bi"].astype(bool)
        ax.scatter(
            x[sig],
            y[sig],
            s=15,
            color=c,
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
        )


def main():
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for label, path in FILES.items():
        if path.exists():
            plot_metric(ax, label, path)
        else:
            print(f"[warn] missing {path}")

    ax.axhline(0, color="0.7", linewidth=1)

    ax.set_xlabel("Hippocampal eigenmode index (k)")
    ax.set_ylabel("Mean standardized β")
    ax.set_title("Language regressors in hippocampal eigenmode space", pad=14)

    ax.legend(frameon=False, fontsize=9, ncol=2)

    ax.text(
        0.99,
        0.03,
        "Dots: q < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    png = OUT_DIR / "figure_hippocampus_all_language_profiles.png"
    pdf = OUT_DIR / "figure_hippocampus_all_language_profiles.pdf"
    svg = OUT_DIR / "figure_hippocampus_all_language_profiles.svg"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ wrote {png}")
    print(f"✅ wrote {pdf}")
    print(f"✅ wrote {svg}")


if __name__ == "__main__":
    main()