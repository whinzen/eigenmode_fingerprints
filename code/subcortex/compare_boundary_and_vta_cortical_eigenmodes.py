#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

VTA_FILE = (
    PANG
    / "subcortex"
    / "vta_subject_cortical_eigenmode_glm_residualized"
    / "vta_to_cortical_eigenmode_residualized_bilateral_group.csv"
)

# Candidate cortical boundary files. The script will use the first existing one.
BOUNDARY_CANDIDATES = [
    PANG / "group_sentence_level_glm" / "group_boundary_allruns.csv",
    PANG / "group_sentence_level_glm" / "group_boundary_by_mode_subject_level.csv",
    PANG / "group_sentence_level_glm" / "group_sentence_boundary_by_mode_subject_level.csv",
]

OUT = PANG / "subcortex" / "vta_cortical_eigenmode_comparison"
OUT.mkdir(parents=True, exist_ok=True)


def find_boundary_file():
    for f in BOUNDARY_CANDIDATES:
        if f.exists():
            return f
    raise FileNotFoundError(
        "Could not find a boundary eigenmode file. Checked:\n"
        + "\n".join(str(f) for f in BOUNDARY_CANDIDATES)
    )


def infer_boundary_beta_column(df):
    candidates = [
        "beta_mean",
        "mean_beta",
        "beta",
        "estimate",
        "coef",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer boundary beta column. Columns: {df.columns.tolist()}")


def infer_mode_column(df):
    candidates = ["mode_k", "mode", "k"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer mode column. Columns: {df.columns.tolist()}")


def main():
    boundary_file = find_boundary_file()

    vta = pd.read_csv(VTA_FILE)
    bnd = pd.read_csv(boundary_file)

    print(f"Using VTA file: {VTA_FILE}")
    print(f"Using boundary file: {boundary_file}")
    print("\nBoundary columns:")
    print(bnd.columns.tolist())

    vta_mode_col = infer_mode_column(vta)
    bnd_mode_col = infer_mode_column(bnd)
    bnd_beta_col = infer_boundary_beta_column(bnd)

    vta = vta.rename(columns={vta_mode_col: "mode_k"})
    bnd = bnd.rename(columns={bnd_mode_col: "mode_k", bnd_beta_col: "boundary_beta"})

    # If boundary file has hemisphere rows, average across hemispheres.
    keep_cols = ["mode_k", "boundary_beta"]
    bnd_small = (
        bnd[keep_cols]
        .groupby("mode_k", as_index=False)
        .mean()
    )

    vta_small = vta[[
        "mode_k",
        "beta_mean",
        "beta_sem",
        "p",
        "q",
        "significant_fdr05",
    ]].rename(columns={
        "beta_mean": "vta_beta",
        "beta_sem": "vta_beta_sem",
        "p": "vta_p",
        "q": "vta_q",
        "significant_fdr05": "vta_significant_fdr05",
    })

    merged = pd.merge(
        bnd_small,
        vta_small,
        on="mode_k",
        how="inner",
    ).sort_values("mode_k")

    # Remove NaNs.
    good = np.isfinite(merged["boundary_beta"]) & np.isfinite(merged["vta_beta"])
    m = merged[good].copy()

    r, p = stats.pearsonr(m["boundary_beta"], m["vta_beta"])
    rho, p_s = stats.spearmanr(m["boundary_beta"], m["vta_beta"])

    # Also compare absolute profiles.
    r_abs, p_abs = stats.pearsonr(
        np.abs(m["boundary_beta"]),
        np.abs(m["vta_beta"]),
    )
    rho_abs, p_s_abs = stats.spearmanr(
        np.abs(m["boundary_beta"]),
        np.abs(m["vta_beta"]),
    )

    summary = pd.DataFrame([
        {
            "comparison": "signed_beta_profiles",
            "n_modes": len(m),
            "pearson_r": r,
            "pearson_p": p,
            "spearman_rho": rho,
            "spearman_p": p_s,
        },
        {
            "comparison": "absolute_beta_profiles",
            "n_modes": len(m),
            "pearson_r": r_abs,
            "pearson_p": p_abs,
            "spearman_rho": rho_abs,
            "spearman_p": p_s_abs,
        },
    ])

    merged_file = OUT / "boundary_vta_cortical_eigenmode_profile_merged.csv"
    summary_file = OUT / "boundary_vta_cortical_eigenmode_profile_correlation.csv"

    merged.to_csv(merged_file, index=False)
    summary.to_csv(summary_file, index=False)

    print(f"\nWrote {merged_file}")
    print(f"Wrote {summary_file}")

    print("\nSummary:")
    print(summary)

    print("\nFirst 20 merged rows:")
    print(merged.head(20))


if __name__ == "__main__":
    main()