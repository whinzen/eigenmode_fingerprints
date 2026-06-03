#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

INCSV = (
    PANG
    / "subcortex"
    / "hippocampus_glm"
    / "joint_boundary_shift"
    / "all_joint_boundary_shift_hipp_glm_rows.csv"
)

OUT = (
    PANG
    / "subcortex"
    / "hippocampus_group"
)
OUT.mkdir(parents=True, exist_ok=True)


def aggregate_effect(df, beta_col, prefix):

    subj = (
        df.groupby(["sub", "hemi", "mode_k"], as_index=False)
        .agg(beta_subject=(beta_col, "mean"))
    )

    rows = []

    for (hemi, k), g in subj.groupby(["hemi", "mode_k"]):

        vals = g["beta_subject"].dropna().values

        n = len(vals)

        if n < 3:
            continue

        mean = np.mean(vals)
        sem = stats.sem(vals)

        tval, pval = stats.ttest_1samp(vals, 0.0)

        rows.append({
            "effect": prefix,
            "hemi": hemi,
            "mode_k": int(k),
            "n_subjects": n,
            "beta_mean": mean,
            "beta_sem": sem,
            "t": tval,
            "p": pval,
        })

    out = pd.DataFrame(rows)

    out["q"] = np.nan
    out["significant_fdr05"] = False

    for hemi in sorted(out["hemi"].unique()):

        idx = out["hemi"] == hemi

        p = out.loc[idx, "p"].values

        sig, q, _, _ = multipletests(
            p,
            alpha=0.05,
            method="fdr_bh",
        )

        out.loc[idx, "q"] = q
        out.loc[idx, "significant_fdr05"] = sig

    return out


def make_bihemi(df):

    L = df[df["hemi"] == "L"].copy()
    R = df[df["hemi"] == "R"].copy()

    bi = L.merge(
        R,
        on=["effect", "mode_k"],
        suffixes=("_L", "_R"),
    )

    bi["beta_mean"] = (
        bi["beta_mean_L"]
        + bi["beta_mean_R"]
    ) / 2.0

    bi["beta_sem"] = np.sqrt(
        bi["beta_sem_L"] ** 2
        + bi["beta_sem_R"] ** 2
    ) / 2.0

    bi["t_bi"] = (
        bi["beta_mean"] / bi["beta_sem"]
    )

    bi["p_bi"] = 2 * (
        1
        - stats.norm.cdf(
            np.abs(bi["t_bi"])
        )
    )

    _, q, _, sig = multipletests(
        bi["p_bi"],
        alpha=0.05,
        method="fdr_bh",
    )

    bi["q_bi"] = q
    bi["significant_fdr05_bi"] = sig

    return bi


def main():

    df = pd.read_csv(INCSV)

    boundary = aggregate_effect(
        df,
        "beta_boundary",
        "boundary_residualized",
    )

    shift = aggregate_effect(
        df,
        "beta_shift",
        "shift_residualized",
    )

    boundary_bi = make_bihemi(boundary)
    shift_bi = make_bihemi(shift)

    out1 = (
        OUT
        / "group_boundary_residualized_hipp_bihemi.csv"
    )

    out2 = (
        OUT
        / "group_shift_residualized_hipp_bihemi.csv"
    )

    boundary_bi.to_csv(out1, index=False)
    shift_bi.to_csv(out2, index=False)

    print("\n=== BOUNDARY RESIDUALIZED ===")
    print(boundary_bi.head())

    print("\n=== SHIFT RESIDUALIZED ===")
    print(shift_bi.head())

    print(f"\n✅ wrote {out1}")
    print(f"✅ wrote {out2}")


if __name__ == "__main__":
    main()