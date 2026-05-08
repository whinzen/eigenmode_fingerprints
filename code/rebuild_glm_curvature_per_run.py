#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist

from settings import PANG_OUT, HEMIS, MIN_TRS

BASE = Path(PANG_OUT)
REG_DIR = BASE / "regressors" / "curvature_per_subject"

# short output name -> actual saved regressor suffix
METRIC_FILES = {
    "global": "global_curvature_R",
    "mean": "mean_turning_angle",
    "path": "path_length",
    "chord": "chord_length",
}


def get_subject_runs(sub_dir: Path):
    runs = []
    for rdir in sorted(sub_dir.glob("run-*")):
        if not rdir.is_dir():
            continue
        try:
            runs.append(int(rdir.name.split("-")[1]))
        except Exception:
            continue
    return sorted(runs)


def load_energy(sub_dir: Path, run: int, hemi: str):
    a_path = sub_dir / f"run-{run:02d}" / f"A_{hemi}.npy"
    if not a_path.exists():
        print(f"[skip] {sub_dir.name} run-{run:02d} hemi-{hemi}: missing {a_path}")
        return None
    A = np.load(a_path)
    if A.ndim != 2:
        print(f"[skip] {sub_dir.name} run-{run:02d} hemi-{hemi}: bad shape {A.shape}")
        return None
    return A ** 2  # [K x T]


def fit_ols_beta_p(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)

    good = np.isfinite(y) & np.isfinite(x)
    y = y[good]
    x = x[good]

    n = len(y)
    if n < 5:
        return np.nan, np.nan

    X = np.column_stack([np.ones(n), x])

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    yhat = X @ beta
    resid = y - yhat

    dof = n - X.shape[1]
    if dof <= 0:
        return float(beta[1]), np.nan

    sse = np.sum(resid ** 2)
    sigma2 = sse / dof

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return float(beta[1]), np.nan

    se_beta1 = np.sqrt(sigma2 * XtX_inv[1, 1])
    if not np.isfinite(se_beta1) or se_beta1 == 0:
        return float(beta[1]), np.nan

    tval = beta[1] / se_beta1
    pval = 2 * t_dist.sf(np.abs(tval), df=dof)
    return float(beta[1]), float(pval)


def main():
    subjects = sorted([p for p in BASE.glob("sub-*") if p.is_dir()])

    for metric_short, metric_suffix in METRIC_FILES.items():
        print(f"\n=== Metric: {metric_short} ({metric_suffix}) ===")

        for sub_dir in subjects:
            sub = sub_dir.name
            out_dir = sub_dir / f"glm_{metric_short}"
            out_dir.mkdir(parents=True, exist_ok=True)

            wide_rows = []
            runs = get_subject_runs(sub_dir)

            for run in runs:
                reg_path = REG_DIR / f"{sub}_run-{run:02d}_{metric_suffix}.npy"
                if not reg_path.exists():
                    print(f"[skip] {sub} run-{run:02d}: missing {reg_path.name}")
                    continue

                predictor = np.load(reg_path).astype(float)

                for hemi in HEMIS:
                    E = load_energy(sub_dir, run, hemi)
                    if E is None:
                        continue

                    K, T_e = E.shape
                    T_x = len(predictor)

                    if T_e != T_x:
                        T = min(T_e, T_x)
                        print(
                            f"[trim] {sub} run-{run:02d} hemi-{hemi}: "
                            f"energy T={T_e}, regressor T={T_x} -> using T={T}"
                        )
                        E = E[:, :T]
                        x = predictor[:T]
                    else:
                        T = T_e
                        x = predictor

                    if T < MIN_TRS:
                        print(f"[skip] {sub} run-{run:02d} hemi-{hemi}: only {T} TRs")
                        continue

                    betas = []
                    pvals = []

                    for k in range(K):
                        beta, p = fit_ols_beta_p(E[k], x)
                        betas.append(beta)
                        pvals.append(p)

                        wide_rows.append({
                            "subject": sub,
                            "run": run,
                            "hemi": hemi,
                            "mode_k": k,
                            "beta": beta,
                            "p": p,
                            "metric": metric_short,
                        })

                    per_run_df = pd.DataFrame({
                        "subject": sub,
                        "run": run,
                        "hemi": hemi,
                        "mode_k": np.arange(K),
                        "beta": betas,
                        "p": pvals,
                        "metric": metric_short,
                    })

                    out_csv = out_dir / f"{metric_short}_{hemi}_run-{run:02d}.csv"
                    per_run_df.to_csv(out_csv, index=False)

            if len(wide_rows) == 0:
                print(f"[skip write empty] {sub} {metric_short}")
                continue

            wide_df = pd.DataFrame(wide_rows)
            wide_out = out_dir / f"{metric_short}_wide.csv"
            wide_df.to_csv(wide_out, index=False)
            print(f"✅ wrote {wide_out}")


if __name__ == "__main__":
    main()