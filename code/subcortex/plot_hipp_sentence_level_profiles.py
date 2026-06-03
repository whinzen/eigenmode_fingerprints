#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN_DIR = PANG / "subcortex" / "hippocampus_group"
OUT_DIR = PANG / "subcortex" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY = IN_DIR / "group_boundary_hipp_bihemi_by_mode_subject_level.csv"
SHIFT = IN_DIR / "group_sentence_shift_hipp_bihemi_by_mode_subject_level.csv"


def load_metric(path):
    df = pd.read_csv(path)
    df = df[df["mode_k"] > 0].sort_values("mode_k").copy()
    return df


def plot_line(ax, df, label, color):
    x = df["mode_k"]
    y = df["beta_mean"]
    sem = df["beta_sem"]

    ax.plot(x, y, linewidth=2.4, label=label, color=color)
    ax.fill_between(x, y - sem, y + sem, alpha=0.18, color=color)

    if "significant_fdr05_bi" in df.columns:
        sig = df["significant_fdr05_bi"].astype(bool)
        ax.scatter(
            x[sig], y[sig],
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )


def main():
    boundary = load_metric(BOUNDARY)
    shift = load_metric(SHIFT)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    plot_line(ax, boundary, "Sentence boundary", "#1f77b4")
    plot_line(ax, shift, "Sentence shift", "#d62728")

    ax.axhline(0, color="0.7", linewidth=1)

    ax.set_xlabel("Hippocampal eigenmode index (k)")
    ax.set_ylabel("Mean standardized β")
    ax.set_title("Sentence-level regressors in hippocampal eigenmode space", pad=14)

    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # small note if significance markers are present
    ax.text(
        0.99, 0.03,
        "Dots: q < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )

    fig.tight_layout()

    png = OUT_DIR / "figure_hippocampus_sentence_level_profiles.png"
    pdf = OUT_DIR / "figure_hippocampus_sentence_level_profiles.pdf"
    svg = OUT_DIR / "figure_hippocampus_sentence_level_profiles.svg"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ wrote {png}")
    print(f"✅ wrote {pdf}")
    print(f"✅ wrote {svg}")


if __name__ == "__main__":
    main()