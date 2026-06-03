#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

ENERGY_DIR = PANG / "subcortex" / "hippocampus_energy"

BOUNDARY_DIR = (
    PANG / "regressors" / "sentence_boundary_hrf_per_subject"
)

SHIFT_DIR = (
    PANG / "regressors" / "sentence_shift_hrf_per_subject"
)

OUT = (
    PANG
    / "subcortex"
    / "hippocampus_glm"
    / "joint_boundary_shift"
)
OUT.mkdir(parents=True, exist_ok=True)


INDEX_CSV = ENERGY_DIR / "hippocampus_energy_index.csv"


def zscore(x):
    x = np.asarray(x, float)
    sd = np.std(x)
    if sd == 0:
        return np.zeros_like(x)
    return (x - np.mean(x)) / sd


def fit_joint_glm(y, boundary, shift):
    """
    y = b0 + b1*boundary + b2*shift
    """

    X = np.column_stack([
        np.ones(len(y)),
        boundary,
        shift,
    ])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ beta

    n = len(y)
    p = X.shape[1]

    sigma2 = np.sum(resid ** 2) / (n - p)

    cov = sigma2 * np.linalg.inv(X.T @ X)

    se = np.sqrt(np.diag(cov))

    tvals = beta / se

    pvals = 2 * (
        1 - stats.t.cdf(np.abs(tvals), df=n - p)
    )

    return beta, tvals, pvals


def main():

    idx = pd.read_csv(INDEX_CSV)

    rows = []

    for _, r in idx.iterrows():

        sub = r["sub"]
        run = r["run"]
        hemi = r["hemi"]

        energy_file = Path(r["energy_file"])

        boundary_file = (
            BOUNDARY_DIR / f"{sub}_{run}.npy"
        )

        shift_file = (
            SHIFT_DIR / f"{sub}_{run}.npy"
        )

        if not boundary_file.exists():
            continue

        if not shift_file.exists():
            continue

        E = np.load(energy_file)

        boundary = np.load(boundary_file)
        shift = np.load(shift_file)

        n_trs = min(
            E.shape[1],
            len(boundary),
            len(shift),
        )

        E = E[:, :n_trs]

        boundary = zscore(boundary[:n_trs])
        shift = zscore(shift[:n_trs])

        for k in range(E.shape[0]):

            y = zscore(E[k])

            beta, tvals, pvals = fit_joint_glm(
                y,
                boundary,
                shift,
            )

            rows.append({
                "sub": sub,
                "run": run,
                "hemi": hemi,
                "mode_k": k,

                "beta_boundary": beta[1],
                "t_boundary": tvals[1],
                "p_boundary": pvals[1],

                "beta_shift": beta[2],
                "t_shift": tvals[2],
                "p_shift": pvals[2],

                "n_trs": n_trs,
            })

        print(f"✅ {sub} {run} hemi-{hemi}")

    out = pd.DataFrame(rows)

    out_csv = (
        OUT
        / "all_joint_boundary_shift_hipp_glm_rows.csv"
    )

    out.to_csv(out_csv, index=False)

    print(f"\n✅ wrote {out_csv}")
    print(out.head())


if __name__ == "__main__":
    main()