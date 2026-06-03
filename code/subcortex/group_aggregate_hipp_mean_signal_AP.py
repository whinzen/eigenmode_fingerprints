#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN_DIR = PANG / "subcortex" / "hipp_mean_signal_AP"
OUT_DIR = PANG / "subcortex" / "hipp_mean_signal_AP_group"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fdr_correct(pvals):
    pvals = np.asarray(pvals, float)

    q = np.full_like(pvals, np.nan)
    sig = np.zeros_like(pvals, dtype=bool)

    good = np.isfinite(pvals)

    if good.sum():
        sig_g, q_g, _, _ = multipletests(
            pvals[good],
            method="fdr_bh",
        )
        q[good] = q_g
        sig[good] = sig_g

    return q, sig


def aggregate_predictor(csv_file):

    predictor = csv_file.stem.replace("mean_signal_AP_", "")

    df = pd.read_csv(csv_file)

    subj = (
        df.groupby(
            ["subject", "hemi", "parcel"],
            as_index=False
        )
        .agg(
            beta_subject=("beta", "mean"),
            n_runs=("run", "nunique")
        )
    )

    rows = []

    for (hemi, parcel), g in subj.groupby(["hemi", "parcel"]):

        vals = g["beta_subject"].values

        beta_mean = np.mean(vals)
        beta_sem = stats.sem(vals)

        tval, pval = stats.ttest_1samp(vals, 0)

        rows.append({
            "predictor": predictor,
            "hemi": hemi,
            "parcel": parcel,
            "n_subjects": len(vals),
            "beta_mean": beta_mean,
            "beta_sem": beta_sem,
            "t": tval,
            "p": pval,
            "mean_runs_per_subject": g["n_runs"].mean(),
        })

    out = pd.DataFrame(rows)

    q, sig = fdr_correct(out["p"].values)

    out["q"] = q
    out["significant_fdr05"] = sig

    out_csv = (
        OUT_DIR /
        f"group_{predictor}_AP_by_hemi.csv"
    )

    out.to_csv(out_csv, index=False)

    print(f"\n{predictor}")
    print(out)

    return out


def main():

    files = sorted(
        IN_DIR.glob("mean_signal_AP_*.csv")
    )

    all_rows = []

    for f in files:
        all_rows.append(
            aggregate_predictor(f)
        )

    all_df = pd.concat(all_rows)

    combined_csv = (
        OUT_DIR /
        "group_all_predictors_AP_by_hemi.csv"
    )

    all_df.to_csv(combined_csv, index=False)

    print("\nWrote:")
    print(combined_csv)


if __name__ == "__main__":
    main()