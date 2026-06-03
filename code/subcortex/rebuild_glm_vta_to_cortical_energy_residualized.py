#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import nibabel as nib

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

ENERGY_DIR = PANG / "energy"
VTA_INDEX = PANG / "subcortex" / "brainstem_roi_timeseries" / "brainstem_roi_timeseries_index.csv"

OUT = PANG / "subcortex" / "vta_cortical_eigenmode_glm_residualized"
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

    X = np.column_stack([
        np.ones(good.sum()),
        nuisance[good],
    ])

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

    try:
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

    except Exception:
        return np.nan, np.nan, np.nan, len(y)


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


def compute_global_bold(bold_file):
    bold = nib.load(str(bold_file)).get_fdata()
    return np.nanmean(
        bold.reshape(-1, bold.shape[-1]),
        axis=0,
    )


def main():
    idx = pd.read_csv(VTA_INDEX)
    idx = idx[idx["roi"].isin(["VTA_L", "VTA_R"])].copy()

    global_cache = {}
    rows = []

    for _, r in idx.iterrows():
        sub = r["subject"]
        run = int(r["run"])
        vta_roi = r["roi"]

        vta = np.load(r["timeseries_file"]).squeeze().astype(float)

        # Whole-brain global BOLD for VTA residualization.
        bold_file = r.get("bold_file", None)
        if bold_file is not None and isinstance(bold_file, str) and Path(bold_file).exists():
            if bold_file not in global_cache:
                global_cache[bold_file] = compute_global_bold(bold_file)
            global_bold = global_cache[bold_file]
            n_vta = min(len(vta), len(global_bold))
            vta_resid = residualize(vta[:n_vta], global_bold[:n_vta])
        else:
            # Fallback: no global signal available in index.
            # This still residualizes cortical energy but uses raw VTA.
            vta_resid = vta.copy()

        for hemi in ["L", "R"]:
            efile = find_energy_file(hemi, run)

            if efile is None:
                print(f"[skip] missing cortical energy: hemi={hemi} run-{run:02d}")
                continue

            E = np.load(efile).astype(float)

            if E.ndim != 2:
                print(f"[skip] bad shape {efile}: {E.shape}")
                continue

            K, T = E.shape

            # Common cortical energy envelope across modes at each TR.
            global_energy = np.nanmean(E, axis=0)

            n = min(T, len(vta_resid))
            x = zscore(vta_resid[:n])
            ge = global_energy[:n]

            for k in range(K):
                # Remove common energy envelope from each mode.
                y_resid = residualize(E[k, :n], ge)
                y = zscore(y_resid)

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
                    "vta_global_residualized": bool(
                        bold_file is not None
                        and isinstance(bold_file, str)
                        and Path(bold_file).exists()
                    ),
                    "energy_global_residualized": True,
                })

            print(
                f"✅ {sub} run-{run:02d} {vta_roi} cortex-{hemi} "
                f"(K={K}, T={T}, n={n})"
            )

    out = pd.DataFrame(rows)

    out_csv = OUT / "vta_to_cortical_eigenmode_residualized_glm_by_run.csv"
    out.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(out.head())


if __name__ == "__main__":
    main()