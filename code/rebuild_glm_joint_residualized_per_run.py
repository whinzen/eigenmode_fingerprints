import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

# =========================
# Paths
# =========================
BASE = Path.home() / "eigenmode_fingerprints"
FUNC_DIR = BASE / "data" / "empirical"
PANG_OUT = BASE / "pang_out"
REG_BASE = PANG_OUT / "regressors"

# =========================
# Helpers
# =========================
def parse_run(fname: str) -> int:
    m = re.search(r"_run-(\d+)_", fname)
    if m is None:
        raise ValueError(f"Could not parse run from: {fname}")
    return int(m.group(1))

def load_mode_energy(subj: str, run: int, hemi: str):
    f = PANG_OUT / subj / f"run-{run}" / f"A_{hemi}.npy"
    if not f.exists():
        return None
    A = np.load(f)
    if A.ndim != 2:
        return None
    return A ** 2  # mode energy

def zscore_safe(x):
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        return x
    s = x.std()
    if s == 0:
        return x - x.mean()
    return (x - x.mean()) / s

def residualize(y, X):
    """
    Residualize y against columns of X.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)

    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    resid = np.full_like(y, np.nan, dtype=float)

    if good.sum() < 3:
        return resid

    Xg = sm.add_constant(X[good], has_constant="add")
    yg = y[good]
    fit = sm.OLS(yg, Xg).fit()
    resid_good = yg - fit.predict(Xg)
    resid[good] = resid_good
    return resid

def load_regressor(metric: str, subj: str, run: int):
    f = REG_BASE / f"{metric}_hrf_per_subject" / f"{subj}_run-{run}.npy"
    if not f.exists():
        return None
    x = np.load(f)
    if x.ndim != 1:
        return None
    return x.astype(float)

# =========================
# Main subject-level routine
# =========================
def run_joint_glm_for_subject(subj: str, base_metric: str, extra_metric: str):
    """
    Model:
        E_k(t) ~ base_metric + residualized(extra_metric | base_metric)

    This lets us separate:
    - shared variance captured by base_metric
    - unique variance in extra_metric beyond base_metric
    """
    func_subj_dir = FUNC_DIR / subj / "func"
    if not func_subj_dir.exists():
        return

    pair_name = f"{base_metric}__plus__{extra_metric}_resid"
    out_dir = PANG_OUT / subj / f"glm_{pair_name}" / "per_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_rows = []

    lfiles = sorted(func_subj_dir.glob(f"{subj}_task-lppEN_run-*_hemi-L_space-fsaverage5_bold.func.gii"))
    runs = [parse_run(f.name) for f in lfiles]

    for run in sorted(runs):
        x_base = load_regressor(base_metric, subj, run)
        x_extra = load_regressor(extra_metric, subj, run)

        if x_base is None:
            print(f"[skip] {subj} run-{run}: missing {base_metric}")
            continue
        if x_extra is None:
            print(f"[skip] {subj} run-{run}: missing {extra_metric}")
            continue
        if len(x_base) != len(x_extra):
            print(f"[skip] {subj} run-{run}: regressor length mismatch")
            continue

        x_base = zscore_safe(x_base)
        x_extra = zscore_safe(x_extra)

        x_extra_resid = residualize(x_extra, x_base[:, None])
        x_extra_resid = zscore_safe(x_extra_resid)

        for hemi in ["L", "R"]:
            E = load_mode_energy(subj, run, hemi)
            if E is None:
                print(f"[skip] {subj} run-{run} hemi-{hemi}: missing A_{hemi}.npy")
                continue

            K, T = E.shape
            if len(x_base) != T:
                print(f"[skip] {subj} run-{run} hemi-{hemi}: regressor length {len(x_base)} != T {T}")
                continue

            X = np.column_stack([x_base, x_extra_resid])
            X = sm.add_constant(X, has_constant="add")

            beta_base, p_base = [], []
            beta_extra, p_extra = [], []

            for k in range(K):
                y = E[k].astype(float)
                if not np.all(np.isfinite(y)) or y.std() == 0:
                    beta_base.append(np.nan)
                    p_base.append(np.nan)
                    beta_extra.append(np.nan)
                    p_extra.append(np.nan)
                    continue

                good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                if good.sum() < 5:
                    beta_base.append(np.nan)
                    p_base.append(np.nan)
                    beta_extra.append(np.nan)
                    p_extra.append(np.nan)
                    continue

                try:
                    model = sm.OLS(y[good], X[good], missing="drop").fit()
                    # params: const, base, extra_resid
                    beta_base.append(model.params[1] if len(model.params) > 1 else np.nan)
                    p_base.append(model.pvalues[1] if len(model.pvalues) > 1 else np.nan)
                    beta_extra.append(model.params[2] if len(model.params) > 2 else np.nan)
                    p_extra.append(model.pvalues[2] if len(model.pvalues) > 2 else np.nan)
                except Exception:
                    beta_base.append(np.nan)
                    p_base.append(np.nan)
                    beta_extra.append(np.nan)
                    p_extra.append(np.nan)

            df = pd.DataFrame({
                "subject": subj,
                "run": run,
                "hemi": hemi,
                "mode_k": np.arange(K, dtype=int),
                f"beta_{base_metric}": beta_base,
                f"p_{base_metric}": p_base,
                f"beta_{extra_metric}_resid": beta_extra,
                f"p_{extra_metric}_resid": p_extra,
            })

            out_csv = out_dir / f"{pair_name}_{hemi}_run-{run}.csv"
            df.to_csv(out_csv, index=False)

            for _, row in df.iterrows():
                wide_rows.append({
                    "subject": subj,
                    "run": run,
                    "hemi": hemi,
                    "mode_k": int(row["mode_k"]),
                    f"beta_{base_metric}": row[f"beta_{base_metric}"],
                    f"p_{base_metric}": row[f"p_{base_metric}"],
                    f"beta_{extra_metric}_resid": row[f"beta_{extra_metric}_resid"],
                    f"p_{extra_metric}_resid": row[f"p_{extra_metric}_resid"],
                })

    wide = pd.DataFrame(wide_rows)
    wide_out = PANG_OUT / subj / f"glm_{pair_name}" / f"{pair_name}_wide.csv"
    wide_out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(wide_out, index=False)
    print(f"[{subj}] wrote {wide_out}")

# =========================
# Entry point
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_metric", required=True, help="e.g. shift")
    ap.add_argument("--extra_metric", required=True, help="e.g. pred_error_ar")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    subjects = sorted([p.name for p in FUNC_DIR.glob("sub-*") if p.is_dir()])
    if args.subjects:
        keep = set(args.subjects)
        subjects = [s for s in subjects if s in keep]

    print(f"Running joint residualized GLMs on {len(subjects)} subjects")
    print(f"Base metric:  {args.base_metric}")
    print(f"Extra metric: {args.extra_metric} (residualized against {args.base_metric})")

    for subj in subjects:
        run_joint_glm_for_subject(subj, args.base_metric, args.extra_metric)

if __name__ == "__main__":
    main()