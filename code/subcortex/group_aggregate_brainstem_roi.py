#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN = PANG / "subcortex" / "brainstem_roi_glm"
OUT = PANG / "subcortex" / "brainstem_roi_group"
OUT.mkdir(parents=True, exist_ok=True)


def fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    sig = np.zeros_like(p, dtype=bool)
    good = np.isfinite(p)

    if good.sum():
        sig_g, q_g, _, _ = multipletests(p[good], method="fdr_bh")
        q[good] = q_g
        sig[good] = sig_g

    return q, sig


def main():
    files = sorted(IN.glob("brainstem_roi_glm_*.csv"))
    all_rows = []

    for f in files:
        df = pd.read_csv(f)

        predictor = df["predictor"].iloc[0]

        subj = (
            df.groupby(["subject", "roi"], as_index=False)
            .agg(
                beta_subject=("beta", "mean"),
                n_runs=("run", "nunique"),
            )
        )

        for roi, g in subj.groupby("roi"):
            vals = g["beta_subject"].dropna().values

            tval, pval = stats.ttest_1samp(vals, 0.0)

            all_rows.append({
                "predictor": predictor,
                "roi": roi,
                "n_subjects": len(vals),
                "beta_mean": vals.mean(),
                "beta_sem": stats.sem(vals),
                "t": tval,
                "p": pval,
                "mean_runs_per_subject": g["n_runs"].mean(),
            })

    out = pd.DataFrame(all_rows)

    q, sig = fdr(out["p"].values)
    out["q"] = q
    out["significant_fdr05"] = sig

    out_csv = OUT / "group_brainstem_roi_glm_all_predictors.csv"
    out.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv}")
    print(out)


if __name__ == "__main__":
    main()