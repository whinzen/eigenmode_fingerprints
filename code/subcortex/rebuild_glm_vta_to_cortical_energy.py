#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

ENERGY_DIR = PANG / "energy"
VTA_INDEX = PANG / "subcortex" / "brainstem_roi_timeseries" / "brainstem_roi_timeseries_index.csv"

OUT = PANG / "subcortex" / "vta_cortical_eigenmode_glm"
OUT.mkdir(parents=True, exist_ok=True)


def zscore(x):
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan)
    good = np.isfinite(x)
    if good.sum() < 5:
        return y
    sd = np.nanstd(x[good])
    if not np.isfinite(sd) or sd == 0:
        return y
    y[good] = (x[good] - np.nanmean(x[good])) / sd
    return y


def ols_beta(y, x):
    good = np.isfinite(y) & np.isfinite(x)
    y = y[good]
    x = x[good]

    if len(y) < 10 or np.std(x) == 0:
        return np.nan, np.nan, np.nan, len(y)

    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ beta
    dof = len(y) - X.shape[1]

    if dof <= 0:
        return beta[1], np.nan, np.nan, len(y)

    s2 = np.sum(resid ** 2) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[1, 1])

    if not np.isfinite(se) or se == 0:
        return beta[1], np.nan, np.nan, len(y)

    tval = beta[1] / se
    pval = 2 * stats.t.sf(abs(tval), dof)

    return beta[1], tval, pval, len(y)


def find_energy_file(hemi, run):
    r = int(str(run).replace("run-", ""))
    candidates = [
        ENERGY_DIR / hemi / f"run-{r}.npy",
        ENERGY_DIR / hemi / f"run-{r:02d}.npy",
    ]
    for f in candidates:
        if f.exists():
            return f
    return None


def main():
    idx = pd.read_csv(VTA_INDEX)
    idx = idx[idx["roi"].isin(["VTA_L", "VTA_R"])].copy()

    rows = []

    for _, r in idx.iterrows():
        sub = r["subject"]
        run = int(r["run"])
        vta_roi = r["roi"]

        vta = np.load(r["timeseries_file"]).squeeze().astype(float)
        vta_z = zscore(vta)

        for hemi in ["L", "R"]:
            efile = find_energy_file(hemi, run)
            if efile is None:
                print(f"[skip] missing cortical energy: hemi={hemi} run-{run}")
                continue

            E = np.load(efile).astype(float)  # K x T

            if E.ndim != 2:
                print(f"[skip] bad shape {efile}: {E.shape}")
                continue

            K, T = E.shape
            n = min(T, len(vta_z))

            x = vta_z[:n]

            for k in range(K):
                y = zscore(E[k, :n])
                beta, tval, pval, n_used = ols_beta(y, x)

                rows.append({
                    "subject": sub,
                    "run": f"{run:02d}",
                    "vta_roi": vta_roi,
                    "cortex_hemi": hemi,
                    "mode_k": k,
                    "beta": beta,
                    "t": tval,
                    "p": pval,
                    "n_used": n_used,
                    "vta_file": r["timeseries_file"],
                    "energy_file": str(efile),
                })

            print(f"✅ {sub} run-{run:02d} {vta_roi} cortex-{hemi}")

    out = pd.DataFrame(rows)
    out_csv = OUT / "vta_to_cortical_eigenmode_glm_by_run.csv"
    out.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(out.head())


if __name__ == "__main__":
    main()