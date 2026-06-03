#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import nibabel as nib

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

ENERGY_INDEX = (
    PANG
    / "cortical_energy_subject"
    / "cortical_energy_subject_index.csv"
)

VTA_INDEX = (
    PANG
    / "subcortex"
    / "brainstem_roi_timeseries"
    / "brainstem_roi_timeseries_index.csv"
)

OUT = PANG / "subcortex" / "vta_subject_cortical_eigenmode_glm_residualized"
OUT.mkdir(parents=True, exist_ok=True)


def zscore(x):
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(x)

    if good.sum() < 5:
        return y

    sd = np.nanstd(x[good])
    if not np.isfinite(sd) or sd == 0:
        return y

    y[good] = (x[good] - np.nanmean(x[good])) / sd
    return y


def residualize(y, nuisance):
    y = np.asarray(y, float)
    nuisance = np.asarray(nuisance, float)

    n = min(len(y), len(nuisance))
    y = y[:n]
    nuisance = nuisance[:n]

    out = np.full(n, np.nan, dtype=float)
    good = np.isfinite(y) & np.isfinite(nuisance)

    if good.sum() < 10:
        return out

    X = np.column_stack([np.ones(good.sum()), nuisance[good]])
    beta, _, _, _ = np.linalg.lstsq(X, y[good], rcond=None)
    out[good] = y[good] - X @ beta

    return out


def ols_beta(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)

    n = min(len(y), len(x))
    y = y[:n]
    x = x[:n]

    good = np.isfinite(y) & np.isfinite(x)
    y = y[good]
    x = x[good]

    if len(y) < 10 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
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


def compute_global_bold(bold_file):
    bold = nib.load(str(bold_file)).get_fdata()
    return np.nanmean(bold.reshape(-1, bold.shape[-1]), axis=0)


def main():
    vta = pd.read_csv(VTA_INDEX)
    eng = pd.read_csv(ENERGY_INDEX)

    # Exclude truncated/corrupted cortical surface runs.
    # Example: sub-EN059 run-15 has only 7 TRs.
    eng = eng[eng["n_trs"] >= 100].copy()

    vta = vta[vta["roi"].isin(["VTA_L", "VTA_R"])].copy()
    vta["run"] = vta["run"].astype(int).map(lambda x: f"{x:02d}")
    eng["run"] = eng["run"].astype(int).map(lambda x: f"{x:02d}")

    merged = vta.merge(
        eng,
        left_on=["subject", "run"],
        right_on=["subject", "run"],
        how="inner",
        suffixes=("_vta", "_energy"),
    )

    print(f"Merged rows: {len(merged)}")
    print(merged[["subject", "run", "roi", "hemi", "timeseries_file", "energy_file"]].head())

    global_cache = {}
    rows = []

    for _, r in merged.iterrows():
        subject = r["subject"]
        run = r["run"]
        vta_roi = r["roi"]
        cortex_hemi = r["hemi"]

        y_vta = np.load(r["timeseries_file"]).squeeze().astype(float)
        E = np.load(r["energy_file"]).astype(float)  # [K,T]

        if E.ndim != 2:
            print(f"[skip] bad energy shape: {r['energy_file']} {E.shape}")
            continue

        # Residualize VTA against whole-brain global BOLD if available.
        bold_file = r.get("bold_file_vta", None)

        if isinstance(bold_file, str) and Path(bold_file).exists():
            if bold_file not in global_cache:
                global_cache[bold_file] = compute_global_bold(bold_file)
            global_bold = global_cache[bold_file]
            n_vta = min(len(y_vta), len(global_bold))
            vta_resid = residualize(y_vta[:n_vta], global_bold[:n_vta])
            vta_global_residualized = True
        else:
            vta_resid = y_vta.copy()
            vta_global_residualized = False

        K, T = E.shape
        global_energy = np.nanmean(E, axis=0)

        n = min(T, len(vta_resid))
        x = zscore(vta_resid[:n])
        ge = global_energy[:n]

        for k in range(K):
            # Residualize each cortical mode against the common cortical energy envelope.
            mode_resid = residualize(E[k, :n], ge)
            y = zscore(mode_resid)

            beta, tval, pval, n_used = ols_beta(y, x)

            rows.append({
                "subject": subject,
                "run": run,
                "vta_roi": vta_roi,
                "cortex_hemi": cortex_hemi,
                "mode_k": k,
                "beta": beta,
                "t": tval,
                "p": pval,
                "n_used": n_used,
                "vta_file": r["timeseries_file"],
                "energy_file": r["energy_file"],
                "vta_global_residualized": vta_global_residualized,
                "energy_global_residualized": True,
            })

        print(f"✅ {subject} run-{run} {vta_roi} cortex-{cortex_hemi} K={K} n={n}")

    out = pd.DataFrame(rows)
    out_csv = OUT / "vta_to_subject_cortical_eigenmode_residualized_glm_by_run.csv"
    out.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(out.head())


if __name__ == "__main__":
    main()