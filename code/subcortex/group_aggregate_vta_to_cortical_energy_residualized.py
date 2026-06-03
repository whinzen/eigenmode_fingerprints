#!/usr/bin/env python

from pathlib import Path

import numpy as np

import pandas as pd

from scipy import stats

from statsmodels.stats.multitest import multipletests

BASE = Path.home() / "eigenmode_fingerprints"

PANG = BASE / "pang_out"

IN = (

    PANG

    / "subcortex"

    / "vta_cortical_eigenmode_glm_residualized"

    / "vta_to_cortical_eigenmode_residualized_glm_by_run.csv"

)

OUT = PANG / "subcortex" / "vta_subject_cortical_eigenmode_glm_residualized/"

OUT.mkdir(parents=True, exist_ok=True)

def fdr(p):

    p = np.asarray(p, float)

    q = np.full_like(p, np.nan)

    sig = np.zeros_like(p, dtype=bool)

    good = np.isfinite(p)

    if good.sum() > 0:

        sig_g, q_g, _, _ = multipletests(

            p[good],

            method="fdr_bh",

        )

        q[good] = q_g

        sig[good] = sig_g

    return q, sig

def main():

    df = pd.read_csv(IN)

    subj = (

        df.groupby(["subject", "vta_roi", "cortex_hemi", "mode_k"], as_index=False)

        .agg(

            beta_subject=("beta", "mean"),

            n_runs=("run", "nunique"),

        )

    )

    subj_csv = OUT / "vta_to_cortical_eigenmode_residualized_subject.csv"

    subj.to_csv(subj_csv, index=False)

    rows = []

    for (vta_roi, cortex_hemi, mode_k), g in subj.groupby(

        ["vta_roi", "cortex_hemi", "mode_k"]

    ):

        vals = g["beta_subject"].dropna().values

        if len(vals) < 3:

            beta_mean = np.nan

            beta_sem = np.nan

            tval = np.nan

            pval = np.nan

        else:

            beta_mean = vals.mean()

            beta_sem = stats.sem(vals)

            tval, pval = stats.ttest_1samp(vals, 0)

        rows.append({

            "vta_roi": vta_roi,

            "cortex_hemi": cortex_hemi,

            "mode_k": int(mode_k),

            "n_subjects": len(vals),

            "beta_mean": beta_mean,

            "beta_sem": beta_sem,

            "t": tval,

            "p": pval,

            "mean_runs": g["n_runs"].mean(),

        })

    group = pd.DataFrame(rows)

    group["q"] = np.nan

    group["significant_fdr05"] = False

    for (vta_roi, hemi), idx in group.groupby(["vta_roi", "cortex_hemi"]).groups.items():

        q, sig = fdr(group.loc[idx, "p"].values)

        group.loc[idx, "q"] = q

        group.loc[idx, "significant_fdr05"] = sig

    group_csv = OUT / "vta_to_cortical_eigenmode_residualized_group_by_hemi.csv"

    group.to_csv(group_csv, index=False)

    # Bilateral VTA and bilateral cortex descriptive average.

    bi = (

        subj.groupby(["subject", "mode_k"], as_index=False)

        .agg(

            beta_subject=("beta_subject", "mean"),

            n_runs=("n_runs", "mean"),

        )

    )

    rows = []

    for mode_k, g in bi.groupby("mode_k"):

        vals = g["beta_subject"].dropna().values

        if len(vals) < 3:

            beta_mean = np.nan

            beta_sem = np.nan

            tval = np.nan

            pval = np.nan

        else:

            beta_mean = vals.mean()

            beta_sem = stats.sem(vals)

            tval, pval = stats.ttest_1samp(vals, 0)

        rows.append({

            "mode_k": int(mode_k),

            "n_subjects": len(vals),

            "beta_mean": beta_mean,

            "beta_sem": beta_sem,

            "t": tval,

            "p": pval,

            "mean_runs": g["n_runs"].mean(),

        })

    bi_group = pd.DataFrame(rows)

    q, sig = fdr(bi_group["p"].values)

    bi_group["q"] = q

    bi_group["significant_fdr05"] = sig

    bi_csv = OUT / "vta_to_cortical_eigenmode_residualized_bilateral_group.csv"

    bi_group.to_csv(bi_csv, index=False)

    print(f"Wrote {subj_csv}")

    print(f"Wrote {group_csv}")

    print(f"Wrote {bi_csv}")

    print(bi_group.head(20))

if __name__ == "__main__":

    main()