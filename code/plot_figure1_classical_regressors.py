from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_DIR = BASE / "group_sentence_level_glm"

BOUNDARY = GROUP_DIR / "group_boundary_by_mode_subject_level.csv"
SHIFT = GROUP_DIR / "group_sentence_shift_by_mode_subject_level.csv"


def load_metric(path, hemi):
    df = pd.read_csv(path)
    df = df[df["hemi"] == hemi].copy()
    return df[["mode_k", "beta_mean", "beta_sem"]].sort_values("mode_k")


def load_bihemi_shared(metric_path):
    dl = load_metric(metric_path, "L")
    dr = load_metric(metric_path, "R")

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"), how="inner")

    merged["beta_mean"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    merged["beta_sem"] = np.sqrt(
        merged["beta_sem_L"].fillna(0.0) ** 2 +
        merged["beta_sem_R"].fillna(0.0) ** 2
    ) / 2.0

    return merged[["mode_k", "beta_mean", "beta_sem"]].sort_values("mode_k")


def main():
    boundary = load_bihemi_shared(BOUNDARY)
    shift = load_bihemi_shared(SHIFT)

    # restrict to shared modes only
    merged = boundary.merge(
        shift,
        on="mode_k",
        suffixes=("_boundary", "_shift"),
        how="inner"
    )
    merged = merged[merged["mode_k"] > 0].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.plot(
        merged["mode_k"],
        merged["beta_mean_boundary"],
        linewidth=2,
        label="Sentence boundary"
    )
    ax.fill_between(
        merged["mode_k"],
        merged["beta_mean_boundary"] - merged["beta_sem_boundary"],
        merged["beta_mean_boundary"] + merged["beta_sem_boundary"],
        alpha=0.18
    )

    ax.plot(
        merged["mode_k"],
        merged["beta_mean_shift"],
        linewidth=2,
        label="Sentence shift"
    )
    ax.fill_between(
        merged["mode_k"],
        merged["beta_mean_shift"] - merged["beta_sem_shift"],
        merged["beta_mean_shift"] + merged["beta_sem_shift"],
        alpha=0.18
    )

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Eigenmode index (k)")
    ax.set_ylabel("Mean beta")
    ax.set_title("Sentence-level regressors and their eigenmode profiles")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    png = OUT_DIR / "figure1_classical_regressors.png"
    pdf = OUT_DIR / "figure1_classical_regressors.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ wrote {png}")
    print(f"✅ wrote {pdf}")


if __name__ == "__main__":
    main()