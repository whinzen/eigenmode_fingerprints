#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN = PANG / "subcortex" / "hipp_trajectory_mean_covariate"
OUT = PANG / "subcortex" / "hipp_trajectory_mean_covariate_group"
OUT.mkdir(parents=True, exist_ok=True)


def fdr_correct(pvals):
    pvals = np.asarray(pvals, float)
    q = np.full_like(pvals, np.nan)
    sig = np.zeros_like(pvals, dtype=bool)
    good = np.isfinite(pvals)

    if good.sum() > 0:
        sig_g, q_g, _, _ = multipletests(pvals[good], method="fdr_bh")
        q[good] = q_g
        sig[good] = sig_g

    return q, sig


def aggregate_file(in_csv):
    df = pd.read_csv(in_csv)

    # remove accidental empty files
    if df.empty:
        raise RuntimeError(f"Empty input: {in_csv}")

    required = {
        "metric",
        "subject",
        "run",
        "hemi",
        "trajectory_column",
        "beta_metric_control_mean",
        "beta_mean_signal",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_csv}: {missing}")

    metric = df["metric"].iloc[0]
    traj_col = df["trajectory_column"].iloc[0]

    # Average runwise betas within subject × hemisphere
    subj = (
        df.groupby(["metric", "trajectory_column", "subject", "hemi"], as_index=False)
        .agg(
            beta_metric_control_mean_subject=("beta_metric_control_mean", "mean"),
            beta_mean_signal_subject=("beta_mean_signal", "mean"),
            n_runs=("run", "nunique"),
        )
    )

    rows = []

    for hemi, g in subj.groupby("hemi"):
        for beta_col, label in [
            ("beta_metric_control_mean_subject", "token_shift_control_mean"),
            ("beta_mean_signal_subject", "mean_signal_covariate"),
        ]:
            vals = g[beta_col].dropna().values
            n = len(vals)

            if n < 3:
                beta_mean = np.nan
                beta_sem = np.nan
                tval = np.nan
                pval = np.nan
            else:
                beta_mean = vals.mean()
                beta_sem = stats.sem(vals)
                tval, pval = stats.ttest_1samp(vals, 0.0)

            rows.append({
                "metric": metric,
                "trajectory_column": traj_col,
                "effect": label,
                "hemi": hemi,
                "n_subjects": n,
                "beta_mean": beta_mean,
                "beta_sem": beta_sem,
                "t": tval,
                "p": pval,
                "mean_runs_per_subject": g["n_runs"].mean(),
                "min_runs_per_subject": g["n_runs"].min(),
                "max_runs_per_subject": g["n_runs"].max(),
            })

    hemi_out = pd.DataFrame(rows)

    q, sig = fdr_correct(hemi_out["p"].values)
    hemi_out["q"] = q
    hemi_out["significant_fdr05"] = sig

    # Bihemispheric subject average
    bi_subj = (
        subj.groupby(["metric", "trajectory_column", "subject"], as_index=False)
        .agg(
            beta_metric_control_mean_subject=("beta_metric_control_mean_subject", "mean"),
            beta_mean_signal_subject=("beta_mean_signal_subject", "mean"),
            n_hemis=("hemi", "nunique"),
            mean_runs=("n_runs", "mean"),
        )
    )

    bi_rows = []

    for beta_col, label in [
        ("beta_metric_control_mean_subject", "token_shift_control_mean"),
        ("beta_mean_signal_subject", "mean_signal_covariate"),
    ]:
        vals = bi_subj[beta_col].dropna().values
        n = len(vals)

        if n < 3:
            beta_mean = np.nan
            beta_sem = np.nan
            tval = np.nan
            pval = np.nan
        else:
            beta_mean = vals.mean()
            beta_sem = stats.sem(vals)
            tval, pval = stats.ttest_1samp(vals, 0.0)

        bi_rows.append({
            "metric": metric,
            "trajectory_column": traj_col,
            "effect": label,
            "hemi": "bihemi_subject_average",
            "n_subjects": n,
            "beta_mean": beta_mean,
            "beta_sem": beta_sem,
            "t": tval,
            "p": pval,
            "mean_runs_per_subject": bi_subj["mean_runs"].mean(),
            "min_runs_per_subject": bi_subj["mean_runs"].min(),
            "max_runs_per_subject": bi_subj["mean_runs"].max(),
        })

    bi_out = pd.DataFrame(bi_rows)

    q_bi, sig_bi = fdr_correct(bi_out["p"].values)
    bi_out["q"] = q_bi
    bi_out["significant_fdr05"] = sig_bi

    stem = in_csv.stem

    subj_csv = OUT / f"subject_{stem}.csv"
    hemi_csv = OUT / f"group_hemi_{stem}.csv"
    bi_csv = OUT / f"group_bihemi_{stem}.csv"

    subj.to_csv(subj_csv, index=False)
    hemi_out.to_csv(hemi_csv, index=False)
    bi_out.to_csv(bi_csv, index=False)

    print(f"\n✅ {stem}")
    print(f"Wrote {subj_csv}")
    print(f"Wrote {hemi_csv}")
    print(f"Wrote {bi_csv}")
    print(bi_out)

    return hemi_out, bi_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trajectory-column",
        choices=["trajectory_step", "trajectory_speed_z", "all"],
        default="all",
    )
    args = ap.parse_args()

    files = sorted(IN.glob("trajectory_*with_mean_covariate_token_shift.csv"))
    files = [f for f in files if not f.name.startswith("._") and f.stat().st_size > 10]

    if args.trajectory_column != "all":
        files = [f for f in files if f"trajectory_{args.trajectory_column}_" in f.name]

    if not files:
        raise RuntimeError("No valid covariate-analysis CSVs found.")

    all_hemi = []
    all_bi = []

    for f in files:
        hemi, bi = aggregate_file(f)
        all_hemi.append(hemi)
        all_bi.append(bi)

    all_hemi = pd.concat(all_hemi, ignore_index=True)
    all_bi = pd.concat(all_bi, ignore_index=True)

    q, sig = fdr_correct(all_hemi["p"].values)
    all_hemi["q_across_all"] = q
    all_hemi["significant_fdr05_across_all"] = sig

    q_bi, sig_bi = fdr_correct(all_bi["p"].values)
    all_bi["q_across_all"] = q_bi
    all_bi["significant_fdr05_across_all"] = sig_bi

    all_hemi_csv = OUT / "group_all_trajectory_mean_covariate_by_hemi.csv"
    all_bi_csv = OUT / "group_all_trajectory_mean_covariate_bihemi.csv"

    all_hemi.to_csv(all_hemi_csv, index=False)
    all_bi.to_csv(all_bi_csv, index=False)

    print("\nCombined:")
    print(f"Wrote {all_hemi_csv}")
    print(f"Wrote {all_bi_csv}")


if __name__ == "__main__":
    main()