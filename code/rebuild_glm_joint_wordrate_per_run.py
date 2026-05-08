#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist

from settings import PANG_OUT, HEMIS, MIN_TRS

BASE = Path(PANG_OUT)

WORDRATE_DIR = BASE / "regressors" / "wordrate_per_subject"
METRIC_BASE = BASE / "regressors"

# adjust these if your filenames differ
METRIC_DIRS = {
    "shift": METRIC_BASE / "sentence_shift_hrf_per_subject",
    "pred_error_ar": METRIC_BASE / "pred_error_ar_hrf_per_subject",
    "pred_error_subspace": METRIC_BASE / "pred_error_subspace_hrf_per_subject",
    "curvature": METRIC_BASE / "curvature_hrf_per_subject",
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


def load_regressor(path: Path):
    if not path.exists():
        return None
    x = np.load(path).astype(float)
    return np.ravel(x)


def fit_ols_multi(y, X):
    """
    OLS:
        y = b0 + X b + e
    where X is [T x P].
    Returns betas for predictors only (excluding intercept), and their p-values.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)

    if X.ndim == 1:
        X = X[:, None]

    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[good]
    X = X[good]

    n = len(y)
    p = X.shape[1]
    if n < p + 3:
        return np.full(p, np.nan), np.full(p, np.nan)

    Xd = np.column_stack([np.ones(n), X])

    try:
        beta, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan), np.full(p, np.nan)

    yhat = Xd @ beta
    resid = y - yhat

    dof = n - Xd.shape[1]
    if dof <= 0:
        return beta[1:].astype(float), np.full(p, np.nan)

    sse = np.sum(resid ** 2)
    sigma2 = sse / dof

    try:
        XtX_inv = np.linalg.inv(Xd.T @ Xd)
    except np.linalg.LinAlgError:
        return beta[1:].astype(float), np.full(p, np.nan)

    se = np.sqrt(np.diag(sigma2 * XtX_inv))[1:]  # predictors only
    betas = beta[1:].astype(float)

    pvals = np.full(p, np.nan)
    for i in range(p):
        if np.isfinite(se[i]) and se[i] > 0:
            tval = betas[i] / se[i]
            pvals[i] = 2 * t_dist.sf(np.abs(tval), df=dof)

    return betas, pvals


def zscore_safe(x):
    x = np.asarray(x, float)
    s = x.std()
    if s == 0 or not np.isfinite(s):
        return x - x.mean()
    return (x - x.mean()) / s


def align_all(arrs):
    """Trim all 1D/2D time series to common T."""
    Ts = []
    for a in arrs:
        if a is None:
            return None
        Ts.append(a.shape[-1])
    T = min(Ts)
    out = []
    for a in arrs:
        if a.ndim == 1:
            out.append(a[:T])
        else:
            out.append(a[..., :T])
    return out


def main():
    subjects = sorted([p for p in BASE.glob("sub-*") if p.is_dir()])

    for metric_name, metric_dir in METRIC_DIRS.items():
        print(f"\n=== JOINT GLM: wordrate + {metric_name} ===")
        if not metric_dir.exists():
            print(f"[skip metric] missing {metric_dir}")
            continue

        for sub_dir in subjects:
            sub = sub_dir.name
            runs = get_subject_runs(sub_dir)

            out_dir = sub_dir / f"glm_joint_wordrate_{metric_name}"
            out_dir.mkdir(parents=True, exist_ok=True)

            wide_rows = []

            for run in runs:
                wr_path = WORDRATE_DIR / f"{sub}_run-{run:02d}.npy"
                met_path = metric_dir / f"{sub}_run-{run:02d}.npy"

                x_wr = load_regressor(wr_path)
                x_met = load_regressor(met_path)

                if x_wr is None:
                    print(f"[skip] {sub} run-{run:02d}: missing wordrate")
                    continue
                if x_met is None:
                    print(f"[skip] {sub} run-{run:02d}: missing {metric_name}")
                    continue

                x_wr = zscore_safe(x_wr)
                x_met = zscore_safe(x_met)

                for hemi in HEMIS:
                    E = load_energy(sub_dir, run, hemi)
                    if E is None:
                        continue

                    aligned = align_all([E, x_wr, x_met])
                    if aligned is None:
                        continue

                    E_al, wr_al, met_al = aligned
                    K, T = E_al.shape

                    if T < MIN_TRS:
                        print(f"[skip] {sub} run-{run:02d} hemi-{hemi}: only {T} TRs")
                        continue

                    X = np.column_stack([wr_al, met_al])  # [T x 2]

                    betas_wr = []
                    betas_met = []
                    p_wr = []
                    p_met = []

                    for k in range(K):
                        b, p = fit_ols_multi(E_al[k], X)
                        betas_wr.append(b[0])
                        betas_met.append(b[1])
                        p_wr.append(p[0])
                        p_met.append(p[1])

                        wide_rows.append({
                            "subject": sub,
                            "run": run,
                            "hemi": hemi,
                            "mode_k": k,
                            "beta_wordrate": b[0],
                            "beta_metric": b[1],
                            "p_wordrate": p[0],
                            "p_metric": p[1],
                            "metric": metric_name,
                        })

                    per_run_df = pd.DataFrame({
                        "subject": sub,
                        "run": run,
                        "hemi": hemi,
                        "mode_k": np.arange(K),
                        "beta_wordrate": betas_wr,
                        "beta_metric": betas_met,
                        "p_wordrate": p_wr,
                        "p_metric": p_met,
                        "metric": metric_name,
                    })

                    out_csv = out_dir / f"joint_wordrate_{metric_name}_{hemi}_run-{run:02d}.csv"
                    per_run_df.to_csv(out_csv, index=False)

            wide_df = pd.DataFrame(wide_rows)
            wide_out = out_dir / f"joint_wordrate_{metric_name}_wide.csv"
            wide_df.to_csv(wide_out, index=False)
            print(f"✅ wrote {wide_out}")


if __name__ == "__main__":
    main()