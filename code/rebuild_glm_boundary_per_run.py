import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist

BASE = Path.home() / "eigenmode_fingerprints"
PANG_OUT = BASE / "pang_out"
REG_DIR = PANG_OUT / "regressors" / "sentence_boundary_hrf_per_subject"

SUBJECT_DIRS = sorted([p for p in PANG_OUT.glob("sub-*") if p.is_dir()])
HEMIS = ["L", "R"]


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


def load_mode_energy(sub_dir: Path, run: int, hemi: str):
    """
    Expected layout:
        pang_out/sub-EN057/run-15/A_L.npy
        pang_out/sub-EN057/run-15/A_R.npy
    """
    a_path = sub_dir / f"run-{run:02d}" / f"A_{hemi}.npy"
    if not a_path.exists():
        print(f"[skip] {sub_dir.name} run-{run:02d} hemi-{hemi}: missing {a_path}")
        return None

    A = np.load(a_path)
    if A.ndim != 2:
        print(f"[skip] {sub_dir.name} run-{run:02d} hemi-{hemi}: bad A shape {A.shape}")
        return None

    # mode energy
    return A ** 2


def fit_ols_beta_p(y, x):
    """
    Raw OLS with intercept:
        y = b0 + b1*x + e

    Returns:
        beta1, p_value
    """
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

    p = X.shape[1]  # intercept + predictor
    dof = n - p
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
    if not REG_DIR.exists():
        raise FileNotFoundError(f"Missing regressor directory: {REG_DIR}")

    print(f"🔍 Found {len(SUBJECT_DIRS)} subjects")

    for sub_dir in SUBJECT_DIRS:
        sub = sub_dir.name
        out_dir = sub_dir / "glm_boundary"
        out_dir.mkdir(parents=True, exist_ok=True)

        wide_rows = []
        subj_runs = get_subject_runs(sub_dir)

        for run in subj_runs:
            reg_path = REG_DIR / f"{sub}_run-{run:02d}.npy"
            if not reg_path.exists():
                print(f"[skip] {sub} run-{run:02d}: missing boundary regressor {reg_path.name}")
                continue

            x = np.load(reg_path).astype(float)

            for hemi in HEMIS:
                E = load_mode_energy(sub_dir, run, hemi)
                if E is None:
                    continue

                K, T_e = E.shape
                T_x = len(x)

                if T_e != T_x:
                    T = min(T_e, T_x)
                    print(
                        f"[trim] {sub} run-{run:02d} hemi-{hemi}: "
                        f"energy T={T_e}, regressor T={T_x} -> using T={T}"
                    )
                    E = E[:, :T]
                    x_use = x[:T]
                else:
                    T = T_e
                    x_use = x

                if T < 5:
                    print(f"[skip] {sub} run-{run:02d} hemi-{hemi}: too short after alignment (T={T})")
                    continue

                betas = []
                pvals = []

                for k in range(K):
                    beta, p = fit_ols_beta_p(E[k], x_use)
                    betas.append(beta)
                    pvals.append(p)

                    wide_rows.append({
                        "subject": sub,
                        "run": run,
                        "hemi": hemi,
                        "mode_k": k,   # keep current native indexing
                        "beta": beta,
                        "p": p,
                    })

                per_run_df = pd.DataFrame({
                    "subject": sub,
                    "run": run,
                    "hemi": hemi,
                    "mode_k": np.arange(K),
                    "beta": betas,
                    "p": pvals,
                })

                out_csv = out_dir / f"boundary_{hemi}_run-{run:02d}.csv"
                per_run_df.to_csv(out_csv, index=False)

        wide_df = pd.DataFrame(wide_rows)
        wide_out = out_dir / "boundary_wide.csv"
        wide_df.to_csv(wide_out, index=False)
        print(f"[{sub}] wrote {wide_out}")


if __name__ == "__main__":
    main()