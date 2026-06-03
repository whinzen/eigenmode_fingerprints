#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN_BASE = PANG / "subcortex" / "hippocampus_glm"
OUT = PANG / "subcortex" / "hippocampus_group"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = [
    "boundary",
    "sentence_shift",
    "token_shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
    "boundary_high_shift",
    "boundary_low_shift",
    "nonboundary_high_shift",
    "nonboundary_low_shift",
]


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


def analyze_metric(metric):
    in_csv = IN_BASE / metric / f"all_{metric}_hipp_glm_rows.csv"

    if not in_csv.exists():
        print(f"[skip] missing {in_csv}")
        return None

    df = pd.read_csv(in_csv)

    if df.empty:
        print(f"[skip] empty {in_csv}")
        return None

    if "mode_k" not in df.columns or "beta" not in df.columns:
        print(f"[skip] missing columns in {in_csv}")
        return None

    # Mode 0 = constant / mean-like hippocampal mode.
    df0 = df[df["mode_k"] == 0].copy()

    if df0.empty:
        print(f"[skip] no mode_k == 0 for {metric}")
        return None

    # Average across runs within subject and hemisphere.
    subj = (
        df0
        .groupby(["sub", "hemi"], as_index=False)
        .agg(beta_subject=("beta", "mean"))
    )

    rows = []

    for hemi, g in subj.groupby("hemi"):
        vals = g["beta_subject"].dropna().values

        if len(vals) < 3:
            continue

        tval, pval = stats.ttest_1samp(vals, 0.0)

        rows.append({
            "metric": metric,
            "hemi": hemi,
            "n_subjects": int(len(vals)),
            "beta_mean": float(np.mean(vals)),
            "beta_sem": float(stats.sem(vals)),
            "t": float(tval),
            "p": float(pval),
        })

    # Bihemispheric subject-level average.
    wide = subj.pivot_table(
        index="sub",
        columns="hemi",
        values="beta_subject",
    ).reset_index()

    if {"L", "R"}.issubset(wide.columns):
        wide["beta_bihemi"] = wide[["L", "R"]].mean(axis=1)
        vals = wide["beta_bihemi"].dropna().values

        if len(vals) >= 3:
            tval, pval = stats.ttest_1samp(vals, 0.0)

            rows.append({
                "metric": metric,
                "hemi": "bihemi",
                "n_subjects": int(len(vals)),
                "beta_mean": float(np.mean(vals)),
                "beta_sem": float(stats.sem(vals)),
                "t": float(tval),
                "p": float(pval),
            })

    return pd.DataFrame(rows)


def main():
    all_rows = []

    for metric in METRICS:
        res = analyze_metric(metric)

        if res is not None and not res.empty:
            all_rows.append(res)

    if not all_rows:
        raise RuntimeError("No mode-0 results found.")

    out = pd.concat(all_rows, ignore_index=True)

    # FDR across all tests.
    q, sig = fdr_correct(out["p"].values)
    out["q_fdr_all"] = q
    out["significant_fdr05_all"] = sig

    # FDR separately for each hemi group.
    out["q_fdr_by_hemi"] = np.nan
    out["significant_fdr05_by_hemi"] = False

    for hemi, idx in out.groupby("hemi").groups.items():
        qh, sigh = fdr_correct(out.loc[idx, "p"].values)
        out.loc[idx, "q_fdr_by_hemi"] = qh
        out.loc[idx, "significant_fdr05_by_hemi"] = sigh

    out_csv = OUT / "hipp_mode0_mean_signal_glm_summary.csv"
    out.to_csv(out_csv, index=False)

    print(f"✅ wrote {out_csv}")
    print("\nAll mode-0 results:")
    print(out.sort_values("p").to_string(index=False))

    print("\nFDR-significant across all tests:")
    print(out[out["significant_fdr05_all"]].sort_values("p").to_string(index=False))


if __name__ == "__main__":
    main()