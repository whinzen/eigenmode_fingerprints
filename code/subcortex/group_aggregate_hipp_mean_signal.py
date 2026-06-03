#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN_BASE = PANG / "subcortex" / "hipp_mean_signal"
OUT = PANG / "subcortex" / "hipp_mean_signal_group"
OUT.mkdir(parents=True, exist_ok=True)


DEFAULT_FILES = {
    "sentence_boundary": "mean_signal_sentence_boundary.csv",
    "sentence_shift": "mean_signal_sentence_shift.csv",
    "token_shift": "mean_signal_token_shift.csv",
    "pred_error_ar": "mean_signal_pred_error_ar.csv",
    "pred_error_subspace": "mean_signal_pred_error_subspace.csv",
    "curvature": "mean_signal_curvature.csv",
}


def fdr_correct(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
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


def standardize_columns(df):
    """
    Allow both old and new column conventions.
    New mean-signal script uses:
        subject, run, hemi, predictor, beta, t, p
    Some older scripts may use:
        sub instead of subject
    """

    df = df.copy()

    if "subject" not in df.columns and "sub" in df.columns:
        df = df.rename(columns={"sub": "subject"})

    required = {"subject", "run", "hemi", "predictor", "beta"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def aggregate_one(predictor, in_csv=None):
    if in_csv is None:
        if predictor not in DEFAULT_FILES:
            raise ValueError(
                f"No default file known for predictor={predictor}. "
                f"Use --in-csv explicitly."
            )
        in_csv = IN_BASE / DEFAULT_FILES[predictor]
    else:
        in_csv = Path(in_csv)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    df = pd.read_csv(in_csv)
    df = standardize_columns(df)

    if df.empty:
        raise RuntimeError(f"Input file is empty: {in_csv}")

    # Keep only this predictor if file contains multiple predictors.
    df = df[df["predictor"] == predictor].copy()

    if df.empty:
        raise RuntimeError(f"No rows found for predictor={predictor} in {in_csv}")

    # Average runwise betas within subject and hemisphere.
    subj = (
        df.groupby(["subject", "hemi"], as_index=False)
        .agg(
            beta_subject=("beta", "mean"),
            n_runs=("run", "nunique"),
            mean_n_voxels=("n_voxels", "mean") if "n_voxels" in df.columns else ("beta", "size"),
        )
    )

    hemi_rows = []

    for hemi, g in subj.groupby("hemi"):
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
            "predictor": predictor,
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

    hemi_out = pd.DataFrame(hemi_rows)

    # FDR across hemispheres for this predictor.
    qvals, sig = fdr_correct(hemi_out["p"].values)
    hemi_out["q"] = qvals
    hemi_out["significant_fdr05"] = sig

    hemi_csv = OUT / f"group_{predictor}_hipp_mean_signal_by_hemi.csv"
    hemi_out.to_csv(hemi_csv, index=False)

    # Bihemispheric subject-level average: average L/R within each subject first.
    bi_subj = (
        subj.groupby("subject", as_index=False)
        .agg(
            beta_subject=("beta_subject", "mean"),
            n_hemis=("hemi", "nunique"),
            mean_runs=("n_runs", "mean"),
        )
    )

    vals = bi_subj["beta_subject"].dropna().values
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

    bi_out = pd.DataFrame([{
        "predictor": predictor,
        "hemi": "bihemi_subject_average",
        "n_subjects": n,
        "beta_mean": beta_mean,
        "beta_sem": beta_sem,
        "t": tval,
        "p": pval,
        "mean_runs_per_subject": bi_subj["mean_runs"].mean(),
        "min_runs_per_subject": bi_subj["mean_runs"].min(),
        "max_runs_per_subject": bi_subj["mean_runs"].max(),
    }])

    # q will be computed later across predictors if running multiple.
    bi_csv = OUT / f"group_{predictor}_hipp_mean_signal_bihemi.csv"
    bi_out.to_csv(bi_csv, index=False)

    subj_csv = OUT / f"subject_{predictor}_hipp_mean_signal_betas.csv"
    subj.to_csv(subj_csv, index=False)

    print(f"\n✅ {predictor}")
    print(f"Input: {in_csv}")
    print(f"Wrote: {hemi_csv}")
    print(f"Wrote: {bi_csv}")
    print(f"Wrote: {subj_csv}")
    print(hemi_out)
    print(bi_out)

    return hemi_out, bi_out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--predictor",
        choices=["sentence_boundary", "sentence_shift", "token_shift", "all"],
        default="all",
    )

    ap.add_argument(
        "--in-csv",
        default=None,
        help="Optional explicit input CSV. Only valid when --predictor is not all.",
    )

    args = ap.parse_args()

    if args.predictor == "all":
        if args.in_csv is not None:
            raise ValueError("--in-csv cannot be used with --predictor all")

        predictors = [
            "sentence_boundary",
            "sentence_shift",
            "token_shift",
            "pred_error_ar",
            "pred_error_subspace",
            "curvature",
    ]
    else:
        predictors = [args.predictor]

    all_hemi = []
    all_bi = []

    for predictor in predictors:
        hemi_out, bi_out = aggregate_one(
            predictor=predictor,
            in_csv=args.in_csv,
        )
        all_hemi.append(hemi_out)
        all_bi.append(bi_out)

    all_hemi = pd.concat(all_hemi, ignore_index=True)
    all_bi = pd.concat(all_bi, ignore_index=True)

    # FDR across all hemisphere-level tests.
    q_hemi, sig_hemi = fdr_correct(all_hemi["p"].values)
    all_hemi["q_across_all_predictors_hemi"] = q_hemi
    all_hemi["significant_fdr05_across_all_predictors_hemi"] = sig_hemi

    # FDR across bihemispheric predictor tests.
    q_bi, sig_bi = fdr_correct(all_bi["p"].values)
    all_bi["q_across_predictors_bihemi"] = q_bi
    all_bi["significant_fdr05_across_predictors_bihemi"] = sig_bi

    all_hemi_csv = OUT / "group_all_predictors_hipp_mean_signal_by_hemi.csv"
    all_bi_csv = OUT / "group_all_predictors_hipp_mean_signal_bihemi.csv"

    all_hemi.to_csv(all_hemi_csv, index=False)
    all_bi.to_csv(all_bi_csv, index=False)

    print("\nCombined outputs:")
    print(f"Wrote: {all_hemi_csv}")
    print(f"Wrote: {all_bi_csv}")


if __name__ == "__main__":
    main()