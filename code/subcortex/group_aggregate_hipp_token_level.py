#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN_BASE = PANG / "subcortex" / "hippocampus_glm"
OUT = PANG / "subcortex" / "hippocampus_group"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = ["token_shift", "pred_error_ar", "pred_error_subspace", "curvature"]


def fdr_correct(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    sig = np.zeros_like(pvals, dtype=bool)

    good = np.isfinite(pvals)

    if good.sum() > 0:
        sig_good, q_good, _, _ = multipletests(
            pvals[good],
            alpha=alpha,
            method="fdr_bh",
        )
        qvals[good] = q_good
        sig[good] = sig_good

    return qvals, sig


def aggregate_metric(metric):
    in_csv = IN_BASE / metric / f"all_{metric}_hipp_glm_rows.csv"

    if not in_csv.exists():
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)
    df = df[df["mode_k"] > 0].copy()

    subj = (
        df.groupby(["sub", "hemi", "mode_k"], as_index=False)
        .agg(beta_subject=("beta", "mean"))
    )

    hemi_rows = []

    for (hemi, k), g in subj.groupby(["hemi", "mode_k"]):
        vals = g["beta_subject"].dropna().values
        n = len(vals)

        if n < 3:
            beta_mean = np.nan
            beta_sem = np.nan
            tval = np.nan
            pval = np.nan
        else:
            beta_mean = np.mean(vals)
            beta_sem = stats.sem(vals)
            tval, pval = stats.ttest_1samp(vals, 0.0)

        hemi_rows.append({
            "metric": metric,
            "hemi": hemi,
            "mode_k": int(k),
            "n_subjects": n,
            "beta_mean": beta_mean,
            "beta_sem": beta_sem,
            "t": tval,
            "p": pval,
        })

    hemi_out = pd.DataFrame(hemi_rows)
    hemi_out["q"] = np.nan
    hemi_out["significant_fdr05"] = False

    for hemi in sorted(hemi_out["hemi"].dropna().unique()):
        idx = hemi_out["hemi"] == hemi
        qvals, sig = fdr_correct(hemi_out.loc[idx, "p"].values)
        hemi_out.loc[idx, "q"] = qvals
        hemi_out.loc[idx, "significant_fdr05"] = sig

    out_csv = OUT / f"group_{metric}_hipp_by_mode_subject_level.csv"
    hemi_out.to_csv(out_csv, index=False)

    L = hemi_out[hemi_out["hemi"] == "L"].copy()
    R = hemi_out[hemi_out["hemi"] == "R"].copy()

    bi = L.merge(
        R,
        on=["metric", "mode_k"],
        suffixes=("_L", "_R"),
        how="inner",
    )

    bi["beta_mean"] = (bi["beta_mean_L"] + bi["beta_mean_R"]) / 2.0
    bi["beta_sem"] = (bi["beta_sem_L"] + bi["beta_sem_R"]) / 2.0
    bi["n_subjects"] = np.minimum(bi["n_subjects_L"], bi["n_subjects_R"])

    bi["t_bi"] = bi["beta_mean"] / bi["beta_sem"]
    bi["p_bi"] = 2 * stats.t.sf(np.abs(bi["t_bi"]), df=bi["n_subjects"] - 1)

    q_bi, sig_bi = fdr_correct(bi["p_bi"].values)
    bi["q_bi"] = q_bi
    bi["significant_fdr05_bi"] = sig_bi

    bi_out = bi[[
        "metric",
        "mode_k",
        "n_subjects",
        "beta_mean",
        "beta_sem",
        "t_bi",
        "p_bi",
        "q_bi",
        "significant_fdr05_bi",
        "beta_mean_L",
        "beta_mean_R",
        "beta_sem_L",
        "beta_sem_R",
        "p_L",
        "p_R",
        "q_L",
        "q_R",
    ]].copy()

    bi_csv = OUT / f"group_{metric}_hipp_bihemi_by_mode_subject_level.csv"
    bi_out.to_csv(bi_csv, index=False)

    print(f"✅ {metric}: wrote {bi_csv}")
    print(bi_out.head())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metric",
        choices=METRICS + ["all"],
        default="all",
    )
    args = ap.parse_args()

    metrics = METRICS if args.metric == "all" else [args.metric]

    for metric in metrics:
        aggregate_metric(metric)


if __name__ == "__main__":
    main()