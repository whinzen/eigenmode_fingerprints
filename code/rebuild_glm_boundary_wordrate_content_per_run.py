#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist

BASE = Path.home() / "eigenmode_fingerprints"
PANG_OUT = BASE / "pang_out"

BOUNDARY_DIR = PANG_OUT / "regressors" / "sentence_boundary_hrf_per_subject"
WORDRATE_DIR = PANG_OUT / "regressors" / "wordrate_per_subject"
CONTENT_DIR = PANG_OUT / "regressors" / "content_density_hrf_per_subject"

SUBJECT_DIRS = sorted([p for p in PANG_OUT.glob("sub-*") if p.is_dir()])
HEMIS = ["L", "R"]
REGRESSOR_NAMES = ["boundary", "wordrate", "content_density"]


def get_subject_runs(sub_dir: Path):
    runs = []
    for rdir in sorted(sub_dir.glob("run-*")):
        if rdir.is_dir():
            try:
                runs.append(int(rdir.name.split("-")[1]))
            except Exception:
                pass
    return sorted(runs)


def load_mode_energy(sub_dir: Path, run: int, hemi: str):
    a_path = sub_dir / f"run-{run:02d}" / f"A_{hemi}.npy"
    if not a_path.exists():
        print(f"[skip] {sub_dir.name} run-{run:02d} hemi-{hemi}: missing {a_path}")
        return None

    A = np.load(a_path)
    if A.ndim != 2:
        print(f"[skip] {sub_dir.name} run-{run:02d} hemi-{hemi}: bad A shape {A.shape}")
        return None

    return A ** 2


def load_regressor(reg_dir: Path, sub: str, run: int):
    for fname in [f"{sub}_run-{run:02d}.npy", f"{sub}_run-{run}.npy"]:
        f = reg_dir / fname
        if f.exists():
            return np.load(f).astype(float)
    return None


def fit_ols_all_betas(y, X):
    """
    OLS:
        y = b0 + b_boundary*boundary + b_wordrate*wordrate
              + b_content*content + e

    Returns beta and p-value for each non-intercept regressor.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)

    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[good]
    X = X[good]

    n = len(y)
    if n < X.shape[1] + 3:
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)

    Xd = np.column_stack([np.ones(n), X])

    try:
        beta, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
        yhat = Xd @ beta
        resid = y - yhat

        dof = n - Xd.shape[1]
        if dof <= 0:
            return beta[1:], np.full(X.shape[1], np.nan)

        sigma2 = np.sum(resid ** 2) / dof
        XtX_inv = np.linalg.inv(Xd.T @ Xd)

        pvals = []
        for j in range(X.shape[1]):
            idx = j + 1
            se = np.sqrt(sigma2 * XtX_inv[idx, idx])
            if not np.isfinite(se) or se == 0:
                pvals.append(np.nan)
            else:
                tval = beta[idx] / se
                pvals.append(2 * t_dist.sf(np.abs(tval), df=dof))

        return beta[1:], np.asarray(pvals)

    except Exception:
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)


def main():
    for d in [BOUNDARY_DIR, WORDRATE_DIR, CONTENT_DIR]:
        if not d.exists():
            raise FileNotFoundError(f"Missing regressor directory: {d}")

    print("Boundary:", BOUNDARY_DIR)
    print("Wordrate:", WORDRATE_DIR)
    print("Content :", CONTENT_DIR)
    print(f"🔍 Found {len(SUBJECT_DIRS)} subjects")

    for sub_dir in SUBJECT_DIRS:
        sub = sub_dir.name
        out_dir = sub_dir / "glm_boundary_wordrate_content"
        out_dir.mkdir(parents=True, exist_ok=True)

        wide_rows = []

        for run in get_subject_runs(sub_dir):
            boundary = load_regressor(BOUNDARY_DIR, sub, run)
            wordrate = load_regressor(WORDRATE_DIR, sub, run)
            content = load_regressor(CONTENT_DIR, sub, run)

            if boundary is None or wordrate is None or content is None:
                print(f"[skip] {sub} run-{run:02d}: missing boundary/wordrate/content regressor")
                continue

            for hemi in HEMIS:
                E = load_mode_energy(sub_dir, run, hemi)
                if E is None:
                    continue

                K, T_e = E.shape
                T = min(T_e, len(boundary), len(wordrate), len(content))

                if T < 5:
                    print(f"[skip] {sub} run-{run:02d} hemi-{hemi}: too short after alignment")
                    continue

                E_use = E[:, :T]

                X = np.column_stack([
                    boundary[:T],
                    wordrate[:T],
                    content[:T],
                ])

                per_run_rows = []

                for k in range(K):
                    betas, pvals = fit_ols_all_betas(E_use[k], X)

                    row = {
                        "subject": sub,
                        "run": run,
                        "hemi": hemi,
                        "mode_k": k,
                        "model": "boundary_plus_wordrate_content",
                    }

                    for name, b, p in zip(REGRESSOR_NAMES, betas, pvals):
                        row[f"beta_{name}"] = b
                        row[f"p_{name}"] = p

                    # compatibility columns: primary target is boundary beta
                    row["beta"] = row["beta_boundary"]
                    row["p"] = row["p_boundary"]

                    per_run_rows.append(row)
                    wide_rows.append(row)

                per_run_df = pd.DataFrame(per_run_rows)
                out_csv = out_dir / f"boundary_wordrate_content_{hemi}_run-{run:02d}.csv"
                per_run_df.to_csv(out_csv, index=False)

        wide_df = pd.DataFrame(wide_rows)
        wide_out = out_dir / "boundary_wordrate_content_wide.csv"
        wide_df.to_csv(wide_out, index=False)
        print(f"[{sub}] wrote {wide_out}")


if __name__ == "__main__":
    main()