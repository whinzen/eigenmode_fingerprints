#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = Path.home() / "eigenmode_fingerprints"
TRAJ = BASE / "pang_out" / "subcortex" / "hippocampus_trajectory"

INFILE = TRAJ / "boundary_token_shift_condition_locked_trajectory_timeseries.csv"
OUTFILE = TRAJ / "boundary_token_shift_condition_locked_contrast_curves_subjectlevel.csv"
RUNLEVEL_OUTFILE = TRAJ / "boundary_token_shift_condition_locked_contrast_curves_runlevel.csv"

CONTRASTS = {
    "boundary_high_minus_boundary_low": (
        "boundary_high_shift",
        "boundary_low_shift",
    ),
    "boundary_high_minus_nonboundary_high": (
        "boundary_high_shift",
        "nonboundary_high_shift",
    ),
    "boundary_low_minus_nonboundary_low": (
        "boundary_low_shift",
        "nonboundary_low_shift",
    ),
}


def fdr(pvals):
    pvals = np.asarray(pvals, float)
    qvals = np.full(len(pvals), np.nan)
    sig = np.zeros(len(pvals), dtype=bool)

    good = np.isfinite(pvals)

    if good.sum() > 0:
        sig_good, q_good, _, _ = multipletests(
            pvals[good],
            alpha=0.05,
            method="fdr_bh",
        )
        qvals[good] = q_good
        sig[good] = sig_good

    return qvals, sig


def main():
    if not INFILE.exists():
        raise FileNotFoundError(INFILE)

    df = pd.read_csv(INFILE)

    runlevel_rows = []

    for contrast_name, (cond_a, cond_b) in CONTRASTS.items():
        A = df[df["condition"] == cond_a].copy()
        B = df[df["condition"] == cond_b].copy()

        merged = A.merge(
            B,
            on=["sub", "run", "hemi", "feature", "tau"],
            suffixes=("_A", "_B"),
            how="inner",
        )

        if merged.empty:
            print(f"[warn] empty merge for {contrast_name}")
            continue

        merged["delta"] = merged["value_A"] - merged["value_B"]
        merged["contrast"] = contrast_name

        runlevel_rows.append(
            merged[
                [
                    "contrast",
                    "sub",
                    "run",
                    "hemi",
                    "feature",
                    "tau",
                    "delta",
                    "value_A",
                    "value_B",
                    "n_events_A",
                    "n_events_B",
                    "regressor_file_A",
                    "regressor_file_B",
                ]
            ]
        )

    if not runlevel_rows:
        raise RuntimeError("No contrast rows created. Check condition names and input file.")

    runlevel = pd.concat(runlevel_rows, ignore_index=True)
    runlevel.to_csv(RUNLEVEL_OUTFILE, index=False)

    # Average across runs within subject before group inference.
    subj = (
        runlevel
        .groupby(["contrast", "sub", "hemi", "feature", "tau"], as_index=False)
        .agg(delta_subject=("delta", "mean"))
    )

    rows = []

    for (contrast, hemi, feature, tau), g in subj.groupby(
        ["contrast", "hemi", "feature", "tau"]
    ):
        vals = g["delta_subject"].dropna().values

        if len(vals) < 3:
            continue

        tval, pval = stats.ttest_1samp(vals, 0.0)

        rows.append({
            "contrast": contrast,
            "hemi": hemi,
            "feature": feature,
            "tau": int(tau),
            "n_subjects": int(len(vals)),
            "mean_delta": float(np.mean(vals)),
            "sem_delta": float(stats.sem(vals)),
            "t": float(tval),
            "p": float(pval),
        })

    out = pd.DataFrame(rows)

    out["q"] = np.nan
    out["sig_fdr05"] = False

    for (contrast, hemi, feature), idx in out.groupby(
        ["contrast", "hemi", "feature"]
    ).groups.items():
        qvals, sig = fdr(out.loc[idx, "p"].values)
        out.loc[idx, "q"] = qvals
        out.loc[idx, "sig_fdr05"] = sig

    out.to_csv(OUTFILE, index=False)

    print(f"✅ wrote {RUNLEVEL_OUTFILE}")
    print(f"✅ wrote {OUTFILE}")

    print("\nFDR-significant rows:")
    print(
        out[out["sig_fdr05"]]
        .sort_values(["contrast", "feature", "tau"])
        .head(100)
    )

    print("\nLowest p-values:")
    print(
        out.sort_values("p")
        .head(20)
    )


if __name__ == "__main__":
    main()