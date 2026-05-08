#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

ORIG = PANG / "group_sentence_level_glm" / "group_boundary_by_mode_subject_level.csv"
CTRL = PANG / "group_boundary_wordrate_content_glm" / "group_boundary_wordrate_content_by_mode_subject_level.csv"

OUT_FIG = PANG / "paper_figures"
OUT_TAB = PANG / "paper_tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)


def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def main():
    orig = pd.read_csv(ORIG)
    ctrl = pd.read_csv(CTRL)

    rows = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, hemi in zip(axes, ["L", "R"]):
        o = orig[orig["hemi"] == hemi][["mode_k", "beta_mean", "beta_sem"]]
        c = ctrl[ctrl["hemi"] == hemi][["mode_k", "beta_mean", "beta_sem"]]

        m = o.merge(c, on="mode_k", suffixes=("_orig", "_ctrl"))
        m = m[m["mode_k"] > 0].copy()

        r, p = pearsonr(m["beta_mean_orig"], m["beta_mean_ctrl"])

        rows.append({
            "hemi": hemi,
            "r_original_vs_controlled": r,
            "p": p,
            "n_modes": len(m),
            "mean_abs_diff": np.mean(np.abs(m["beta_mean_orig"] - m["beta_mean_ctrl"])),
        })

        ax.plot(
            m["mode_k"],
            zscore(m["beta_mean_orig"]),
            linewidth=2,
            label="Original boundary model",
        )
        ax.plot(
            m["mode_k"],
            zscore(m["beta_mean_ctrl"]),
            linewidth=2,
            linestyle="--",
            label="Boundary + wordrate + content density",
        )

        ax.axhline(0, linewidth=0.8, alpha=0.4)
        ax.set_title(f"Hemi-{hemi}: r = {r:.6f}")
        ax.set_xlabel("Mode index")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("z-scored boundary β profile")
    axes[0].legend(frameon=False, fontsize=9)

    fig.suptitle("Sentence-boundary eigenmode profiles are preserved after lexical and temporal controls")
    plt.tight_layout()

    out_png = OUT_FIG / "figure_boundary_original_vs_wordrate_content_controlled.png"
    out_pdf = OUT_FIG / "figure_boundary_original_vs_wordrate_content_controlled.pdf"
    out_csv = OUT_TAB / "table_boundary_original_vs_wordrate_content_controlled.csv"

    plt.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    plt.close()

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"✅ wrote {out_png}")
    print(f"✅ wrote {out_pdf}")
    print(f"✅ wrote {out_csv}")
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()