#!/usr/bin/env python

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

# Qwen regressors ONLY
REGRESSOR_DIR = PANG_OUT / "regressors_qwen3_0p6b"

# Per-subject Qwen GLM outputs
QWEN_LABEL = "qwen3_0p6b"


# =========================
# Config
# =========================
METRICS = [
    "shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]


# =========================
# Helpers
# =========================
def parse_run(fname: str) -> int:
    m = re.search(r"_run-(\d+)_", fname)
    if m is None:
        raise ValueError(f"Could not parse run from: {fname}")
    return int(m.group(1))


def load_mode_energy(subj: str, run: int, hemi: str):
    """
    Load A_H.npy, shape (K, T), and convert to energy A^2.
    """
    f = PANG_OUT / subj / f"run-{run}" / f"A_{hemi}.npy"
    if not f.exists():
        return None

    A = np.load(f)

    if A.ndim != 2:
        return None

    return A ** 2


def run_glm_for_subject_metric(subj: str, metric: str):
    func_subj_dir = FUNC_DIR / subj / "func"

    if not func_subj_dir.exists():
        return

    reg_dir = REGRESSOR_DIR / f"{metric}_hrf_per_subject"

    if not reg_dir.exists():
        raise FileNotFoundError(f"Missing regressor folder: {reg_dir}")

    out_dir = PANG_OUT / subj / f"glm_{QWEN_LABEL}_{metric}" / "per_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_rows = []

    lfiles = sorted(
        func_subj_dir.glob(
            f"{subj}_task-lppEN_run-*_hemi-L_space-fsaverage5_bold.func.gii"
        )
    )

    runs = [parse_run(f.name) for f in lfiles]

    for run in sorted(runs):
        reg_path = reg_dir / f"{subj}_run-{run}.npy"

        if not reg_path.exists():
            print(f"[skip] {subj} run-{run}: missing {metric} regressor")
            continue

        x = np.load(reg_path)

        if x.ndim != 1:
            print(f"[skip] {subj} run-{run}: regressor not 1D")
            continue

        for hemi in ["L", "R"]:
            E = load_mode_energy(subj, run, hemi)

            if E is None:
                print(f"[skip] {subj} run-{run} hemi-{hemi}: missing A_{hemi}.npy")
                continue

            K, T = E.shape

            if len(x) != T:
                print(
                    f"[skip] {subj} run-{run} hemi-{hemi}: "
                    f"regressor length {len(x)} != T {T}"
                )
                continue

            X = sm.add_constant(x.astype(float))
            betas, pvals = [], []

            for k in range(K):
                y = E[k].astype(float)

                good = np.isfinite(y) & np.isfinite(x)

                if good.sum() < 5 or np.nanstd(y[good]) == 0:
                    betas.append(np.nan)
                    pvals.append(np.nan)
                    continue

                try:
                    model = sm.OLS(y[good], X[good], missing="drop").fit()
                    betas.append(model.params[1] if len(model.params) > 1 else np.nan)
                    pvals.append(model.pvalues[1] if len(model.pvalues) > 1 else np.nan)
                except Exception:
                    betas.append(np.nan)
                    pvals.append(np.nan)

            df = pd.DataFrame({
                "subject": subj,
                "run": run,
                "hemi": hemi,
                "mode_k": np.arange(K, dtype=int),
                "beta": betas,
                "p": pvals,
                "metric": metric,
                "embedding_model": QWEN_LABEL,
            })

            out_csv = out_dir / f"{metric}_{hemi}_run-{run}.csv"
            df.to_csv(out_csv, index=False)

            for k, b, p in zip(df["mode_k"], df["beta"], df["p"]):
                wide_rows.append({
                    "subject": subj,
                    "run": run,
                    "hemi": hemi,
                    "mode_k": int(k),
                    "beta": b,
                    "p": p,
                    "metric": metric,
                    "embedding_model": QWEN_LABEL,
                })

    wide = pd.DataFrame(wide_rows)
    wide_out = PANG_OUT / subj / f"glm_{QWEN_LABEL}_{metric}" / f"{metric}_wide.csv"
    wide_out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(wide_out, index=False)

    print(f"[{subj}] wrote {wide_out} ({len(wide)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metric",
        default="all",
        choices=METRICS + ["all"],
        help="Metric to run, or all.",
    )
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    metrics = METRICS if args.metric == "all" else [args.metric]

    subjects = sorted([p.name for p in FUNC_DIR.glob("sub-*") if p.is_dir()])

    if args.subjects:
        keep = set(args.subjects)
        subjects = [s for s in subjects if s in keep]

    print(f"Subjects: {len(subjects)}")
    print(f"Metrics: {metrics}")
    print(f"Regressor root: {REGRESSOR_DIR}")

    for metric in metrics:
        print(f"\n=== Metric: {metric} ===")
        for subj in subjects:
            run_glm_for_subject_metric(subj, metric)


if __name__ == "__main__":
    main()