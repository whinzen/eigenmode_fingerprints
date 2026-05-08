#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

OUT_FIG = PANG / "paper_figures"
OUT_TAB = PANG / "paper_tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
METRICS = {
    "shift": "Token shift",
    "pred_error_ar": "Prediction error",
    "pred_error_subspace": "Subspace exit",
    "curvature": "Curvature",
}

HEMIS = ["L", "R"]

# =========================
# Helpers
# =========================
def load_profile(metric, hemi, model):
    if model == "bert":
        f = (
            PANG / f"group_{metric}_glm"
            / f"group_{metric}_hemi-{hemi}_by_mode_subject_level.csv"
        )
    elif model == "qwen":
        f = (
            PANG / f"group_qwen3_0p6b_{metric}_glm"
            / f"group_qwen3_0p6b_{metric}_hemi-{hemi}_by_mode_subject_level.csv"
        )
    else:
        raise ValueError(model)

    if not f.exists():
        raise FileNotFoundError(f)

    df = pd.read_csv(f)

    if "k" in df.columns and "mode_k" not in df.columns:
        df = df.rename(columns={"k": "mode_k"})

    return df.sort_values("mode_k")


def safe_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return np.nan, np.nan
    return pearsonr(x[good], y[good])


def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def cosine_similarity(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# =========================
# Main
# =========================
def main():

    rows = []

    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=2,
        figsize=(10, 9),
        sharex=True,
    )

    for i, (metric, label) in enumerate(METRICS.items()):
        for j, hemi in enumerate(HEMIS):

            ax = axes[i, j]

            b = load_profile(metric, hemi, "bert")
            q = load_profile(metric, hemi, "qwen")

            m = b[["mode_k", "beta_mean"]].merge(
                q[["mode_k", "beta_mean"]],
                on="mode_k",
                suffixes=("_bert", "_qwen"),
            )

            x = m["beta_mean_bert"].values
            y = m["beta_mean_qwen"].values

            # correlations
            r_raw, p_raw = safe_corr(x, y)
            r_s, p_s = spearmanr(x, y)

            # cosine similarity
            cos_sim = cosine_similarity(x, y)

            # slope + intercept
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

            # z-profile correlation (shape comparison)
            r_z, p_z = safe_corr(zscore(x), zscore(y))

            rows.append({
                "metric": metric,
                "label": label,
                "hemi": hemi,
                "pearson_r": r_raw,
                "spearman_r": r_s,
                "cosine_similarity": cos_sim,
                "z_profile_r": r_z,
                "slope": slope,
                "intercept": intercept,
                "n_modes": len(m),
            })

            # plot (z-scored profiles)
            ax.plot(
                m["mode_k"],
                zscore(x),
                label="BERT",
                linewidth=2,
            )

            ax.plot(
                m["mode_k"],
                zscore(y),
                label="Qwen3-0.6B",
                linestyle="--",
                linewidth=2,
            )

            ax.axhline(0, linewidth=0.8, alpha=0.4)

            ax.set_title(
                f"{label}, hemi-{hemi}\n"
                f"r(z)={r_z:.3f}, cos={cos_sim:.3f}",
                fontsize=9,
            )

            if i == len(METRICS) - 1:
                ax.set_xlabel("Mode index")

            if j == 0:
                ax.set_ylabel("z-scored β")

            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()

    fig_path = OUT_FIG / "figure_bert_vs_qwen_profiles.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    table = pd.DataFrame(rows)
    table_path = OUT_TAB / "table_bert_vs_qwen_comparison.csv"
    table.to_csv(table_path, index=False)

    print(f"✅ wrote {fig_path}")
    print(f"✅ wrote {table_path}")
    print("\nSummary:")
    print(table)


if __name__ == "__main__":
    main()