#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

ENERGY_INDEX = (
    PANG
    / "subcortex"
    / "hippocampus_energy"
    / "hippocampus_energy_index.csv"
)

BOUNDARY_DIR = (
    PANG
    / "regressors"
    / "sentence_boundary_hrf_per_subject"
)

OUT = (
    PANG
    / "subcortex"
    / "hippocampus_trajectory"
)
OUT.mkdir(parents=True, exist_ok=True)

K_USE = 30
EXCLUDE_MODE0 = True
BOUNDARY_Q = 0.75
NONBOUNDARY_Q = 0.25


def zscore_rows(X):
    X = np.asarray(X, float)
    out = np.zeros_like(X)

    for i in range(X.shape[0]):
        x = X[i]
        good = np.isfinite(x)
        if good.sum() < 3:
            continue
        mu = np.nanmean(x[good])
        sd = np.nanstd(x[good])
        if sd == 0 or not np.isfinite(sd):
            continue
        out[i, good] = (x[good] - mu) / sd

    return out


def find_boundary_regressor(sub, run):
    run_num = run.replace("run-", "").lstrip("0")
    run_num_2 = f"{int(run_num):02d}"

    candidates = [
        BOUNDARY_DIR / f"{sub}_{run}.npy",
        BOUNDARY_DIR / f"{sub}_run-{run_num}.npy",
        BOUNDARY_DIR / f"{sub}_run-{run_num_2}.npy",
    ]

    for f in candidates:
        if f.exists():
            return f

    hits = list(BOUNDARY_DIR.glob(f"*{sub}*run-{run_num}*.npy"))
    hits += list(BOUNDARY_DIR.glob(f"*{sub}*run-{run_num_2}*.npy"))

    if hits:
        return sorted(hits)[0]

    return None


def compute_features(A):
    """
    A is K x T mode-amplitude matrix.

    Returns:
    - step: ||A(t)-A(t-1)||, length T
    - speed_z: z-scored step, length T
    - angle: angle between successive displacement vectors, length T
    """

    A = np.asarray(A, float)

    if EXCLUDE_MODE0:
        A = A[1:K_USE, :]
    else:
        A = A[:K_USE, :]

    A = zscore_rows(A)

    K, T = A.shape

    step = np.zeros(T, dtype=float)
    angle = np.full(T, np.nan, dtype=float)

    dA = np.diff(A, axis=1)  # K x (T-1)

    step[1:] = np.linalg.norm(dA, axis=0)

    # angle at t compares displacement t-1 -> t and t -> t+1
    for t in range(1, T - 1):
        v1 = dA[:, t - 1]
        v2 = dA[:, t]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 == 0 or n2 == 0:
            continue

        c = np.dot(v1, v2) / (n1 * n2)
        c = np.clip(c, -1.0, 1.0)

        angle[t] = np.arccos(c)

    speed_z = zscore_vector(step)

    return step, speed_z, angle


def zscore_vector(x):
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    good = np.isfinite(x)

    if good.sum() < 3:
        return y

    mu = np.nanmean(x[good])
    sd = np.nanstd(x[good])

    if sd == 0 or not np.isfinite(sd):
        return y

    y[good] = (x[good] - mu) / sd

    return y


def compare_boundary_nonboundary(feature, boundary):
    """
    Compare feature values at high-boundary TRs vs low-boundary TRs.

    Because boundary regressors are HRF-convolved, this uses quantiles:
    - boundary TRs: top quartile of boundary regressor
    - nonboundary TRs: bottom quartile
    """

    n = min(len(feature), len(boundary))
    f = np.asarray(feature[:n], float)
    b = np.asarray(boundary[:n], float)

    good = np.isfinite(f) & np.isfinite(b)

    if good.sum() < 10:
        return None

    f = f[good]
    b = b[good]

    b_hi = np.nanquantile(b, BOUNDARY_Q)
    b_lo = np.nanquantile(b, NONBOUNDARY_Q)

    boundary_mask = b >= b_hi
    nonboundary_mask = b <= b_lo

    fb = f[boundary_mask]
    fn = f[nonboundary_mask]

    fb = fb[np.isfinite(fb)]
    fn = fn[np.isfinite(fn)]

    if len(fb) < 3 or len(fn) < 3:
        return None

    tval, pval = stats.ttest_ind(fb, fn, equal_var=False)

    return {
        "boundary_mean": float(np.mean(fb)),
        "nonboundary_mean": float(np.mean(fn)),
        "boundary_minus_nonboundary": float(np.mean(fb) - np.mean(fn)),
        "t": float(tval),
        "p": float(pval),
        "n_boundary_trs": int(len(fb)),
        "n_nonboundary_trs": int(len(fn)),
    }


def main():
    index = pd.read_csv(ENERGY_INDEX)

    all_time_rows = []
    contrast_rows = []

    for _, r in index.iterrows():

        sub = r["sub"]
        run = r["run"]
        hemi = r["hemi"]

        energy_file = Path(r["energy_file"])
        amp_file = Path(str(energy_file).replace("_energy.npy", "_amplitudes.npy"))

        if not amp_file.exists():
            print(f"[skip] missing amplitudes: {amp_file}")
            continue

        boundary_file = find_boundary_regressor(sub, run)

        if boundary_file is None:
            print(f"[skip] missing boundary regressor: {sub} {run}")
            continue

        A = np.load(amp_file)
        boundary = np.load(boundary_file).squeeze().astype(float)

        step, speed_z, angle = compute_features(A)

        n = min(len(step), len(boundary))

        out_ts = pd.DataFrame({
            "sub": sub,
            "run": run,
            "hemi": hemi,
            "tr": np.arange(n),
            "trajectory_step": step[:n],
            "trajectory_speed_z": speed_z[:n],
            "trajectory_angle": angle[:n],
            "boundary_regressor": boundary[:n],
            "amplitude_file": str(amp_file),
            "boundary_file": str(boundary_file),
        })

        ts_file = OUT / f"{sub}_{run}_hemi-{hemi}_trajectory_features.csv"
        out_ts.to_csv(ts_file, index=False)

        all_time_rows.append(out_ts)

        for feat_name, feat in [
            ("trajectory_step", step),
            ("trajectory_speed_z", speed_z),
            ("trajectory_angle", angle),
        ]:
            comp = compare_boundary_nonboundary(feat, boundary)

            if comp is None:
                continue

            comp.update({
                "sub": sub,
                "run": run,
                "hemi": hemi,
                "feature": feat_name,
                "trajectory_file": str(ts_file),
            })

            contrast_rows.append(comp)

        print(f"✅ {sub} {run} hemi-{hemi}")

    if all_time_rows:
        all_ts = pd.concat(all_time_rows, ignore_index=True)
        all_ts_file = OUT / "all_hipp_trajectory_features_timeseries.csv"
        all_ts.to_csv(all_ts_file, index=False)
        print(f"\n✅ wrote {all_ts_file}")

    contrasts = pd.DataFrame(contrast_rows)
    contrast_file = OUT / "boundary_vs_nonboundary_trajectory_contrasts_runlevel.csv"
    contrasts.to_csv(contrast_file, index=False)

    print(f"✅ wrote {contrast_file}")

    if not contrasts.empty:
        print(
            contrasts
            .groupby(["feature", "hemi"])["boundary_minus_nonboundary"]
            .describe()
        )


if __name__ == "__main__":
    main()