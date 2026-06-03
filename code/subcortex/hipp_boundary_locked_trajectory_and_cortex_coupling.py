#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

TRAJ_CSV = (
    PANG / "subcortex" / "hippocampus_trajectory"
    / "all_hipp_trajectory_features_timeseries.csv"
)

OUT = PANG / "subcortex" / "hippocampus_trajectory"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "trajectory_step",
    "trajectory_speed_z",
    "trajectory_angle",
]

PRE_BOUNDARY_DIR = PANG / "regressors" / "sentence_boundary_per_subject"

WIN_PRE = 5
WIN_POST = 8


def find_boundary_file(sub, run):
    run_num = run.replace("run-", "").lstrip("0")
    run_num_2 = f"{int(run_num):02d}"

    candidates = [
        PRE_BOUNDARY_DIR / f"{sub}_{run}.npy",
        PRE_BOUNDARY_DIR / f"{sub}_run-{run_num}.npy",
        PRE_BOUNDARY_DIR / f"{sub}_run-{run_num_2}.npy",
    ]

    for f in candidates:
        if f.exists():
            return f

    hits = list(PRE_BOUNDARY_DIR.glob(f"*{sub}*run-{run_num}*.npy"))
    hits += list(PRE_BOUNDARY_DIR.glob(f"*{sub}*run-{run_num_2}*.npy"))

    if hits:
        return sorted(hits)[0]

    return None


def valid_event_window(t, n, pre=WIN_PRE, post=WIN_POST):
    return (t - pre >= 0) and (t + post < n)


def event_locked_average(y, events, pre=WIN_PRE, post=WIN_POST):
    y = np.asarray(y, float)
    n = len(y)

    windows = []

    for t in events:
        if valid_event_window(t, n, pre, post):
            windows.append(y[t - pre:t + post + 1])

    if len(windows) == 0:
        return None, 0

    return np.nanmean(np.vstack(windows), axis=0), len(windows)


def choose_matched_nonboundary_events(boundary, n_events, pre=WIN_PRE, post=WIN_POST, seed=1):
    """
    Choose nonboundary TRs from TRs with no sentence boundary impulse.
    Do not exclude a wide temporal neighborhood, because sentence
    boundaries are dense in this naturalistic story.
    """
    rng = np.random.default_rng(seed)

    boundary = np.asarray(boundary, float)

    candidates = np.where(boundary == 0)[0]
    candidates = np.array([
        t for t in candidates
        if valid_event_window(t, len(boundary), pre, post)
    ])

    if len(candidates) == 0:
        return np.array([], dtype=int)

    n_pick = min(n_events, len(candidates))

    return rng.choice(candidates, size=n_pick, replace=False)


def boundary_locked_hc():
    df = pd.read_csv(TRAJ_CSV)

    rows = []
    contrast_rows = []

    taus = np.arange(-WIN_PRE, WIN_POST + 1)

    for (sub, run, hemi), g in df.groupby(["sub", "run", "hemi"]):
        g = g.sort_values("tr")

        bf = find_boundary_file(sub, run)

        if bf is None:
            print(f"[skip] missing pre-HRF boundary: {sub} {run}")
            continue

        boundary = np.load(bf).squeeze().astype(float)

        n = min(len(boundary), len(g))
        boundary = boundary[:n]
        g = g.iloc[:n].copy()

        boundary_events = np.where(boundary > 0)[0]

        if len(boundary_events) < 3:
            continue

        nonboundary_events = choose_matched_nonboundary_events(
            boundary,
            n_events=len(boundary_events),
            seed=abs(hash((sub, run, hemi))) % (2**32),
        )

        if len(nonboundary_events) < 3:
            continue

        for feat in FEATURES:
            y = g[feat].values

            b_avg, n_b = event_locked_average(y, boundary_events)
            nb_avg, n_nb = event_locked_average(y, nonboundary_events)

            if b_avg is None or nb_avg is None:
                continue

            for tau, vb, vn in zip(taus, b_avg, nb_avg):
                rows.append({
                    "sub": sub,
                    "run": run,
                    "hemi": hemi,
                    "feature": feat,
                    "tau": int(tau),
                    "boundary_mean": vb,
                    "nonboundary_mean": vn,
                    "boundary_minus_nonboundary": vb - vn,
                    "n_boundary_events": n_b,
                    "n_nonboundary_events": n_nb,
                })

            # compact run-level contrast at tau = 0
            t0 = np.where(taus == 0)[0][0]

            contrast_rows.append({
                "sub": sub,
                "run": run,
                "hemi": hemi,
                "feature": feat,
                "boundary_minus_nonboundary_tau0": b_avg[t0] - nb_avg[t0],
                "boundary_tau0": b_avg[t0],
                "nonboundary_tau0": nb_avg[t0],
                "n_boundary_events": n_b,
                "n_nonboundary_events": n_nb,
            })

        print(f"✅ {sub} {run} hemi-{hemi}")

    locked = pd.DataFrame(rows)
    contrasts = pd.DataFrame(contrast_rows)

    locked_csv = OUT / "boundary_locked_hipp_trajectory_timeseries.csv"
    contrast_csv = OUT / "boundary_locked_hipp_trajectory_tau0_contrasts.csv"

    locked.to_csv(locked_csv, index=False)
    contrasts.to_csv(contrast_csv, index=False)

    print(f"\n✅ wrote {locked_csv}")
    print(f"✅ wrote {contrast_csv}")

    # Subject-level stats
    subj = (
        contrasts
        .groupby(["sub", "hemi", "feature"], as_index=False)
        .agg(effect=("boundary_minus_nonboundary_tau0", "mean"))
    )

    stat_rows = []

    for (hemi, feat), g in subj.groupby(["hemi", "feature"]):
        vals = g["effect"].dropna().values

        if len(vals) < 3:
            continue

        tval, pval = stats.ttest_1samp(vals, 0.0)

        stat_rows.append({
            "hemi": hemi,
            "feature": feat,
            "n_subjects": len(vals),
            "mean_effect": np.mean(vals),
            "sem": stats.sem(vals),
            "t": tval,
            "p": pval,
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_csv = OUT / "boundary_locked_hipp_trajectory_subject_stats.csv"
    stat_df.to_csv(stat_csv, index=False)

    print(f"✅ wrote {stat_csv}")
    print(stat_df)

    return contrasts


def cortex_hc_coupling(cortex_csv):
    """
    Optional coupling test.

    Expects cortical CSV with columns:
    sub, run, hemi, mode_k, beta

    Uses mean cortical beta across low modes 1..20 as cortical integration index.
    Merges with HC trajectory tau0 contrast.
    """
    hc_csv = OUT / "boundary_locked_hipp_trajectory_tau0_contrasts.csv"

    if not hc_csv.exists():
        raise FileNotFoundError(hc_csv)

    hc = pd.read_csv(hc_csv)
    cx = pd.read_csv(cortex_csv)

    required = {"sub", "run", "hemi", "mode_k", "beta"}
    missing = required.difference(cx.columns)

    if missing:
        raise RuntimeError(f"Cortical CSV missing columns: {missing}")

    cx = cx[(cx["mode_k"] > 0) & (cx["mode_k"] <= 20)].copy()

    cx_low = (
        cx.groupby(["sub", "run", "hemi"], as_index=False)
        .agg(cortical_lowmode_beta=("beta", "mean"))
    )

    hc_step = hc[hc["feature"] == "trajectory_speed_z"].copy()

    merged = hc_step.merge(
        cx_low,
        on=["sub", "run", "hemi"],
        how="inner",
    )

    out_csv = OUT / "cortex_lowmode_vs_hipp_mobility_merged.csv"
    merged.to_csv(out_csv, index=False)

    print(f"\n✅ wrote {out_csv}")

    subj = (
        merged
        .groupby(["sub", "hemi"], as_index=False)
        .agg(
            cortical_lowmode_beta=("cortical_lowmode_beta", "mean"),
            hipp_mobility_effect=("boundary_minus_nonboundary_tau0", "mean"),
        )
    )

    rows = []

    for hemi, g in subj.groupby("hemi"):
        x = g["cortical_lowmode_beta"].values
        y = g["hipp_mobility_effect"].values

        good = np.isfinite(x) & np.isfinite(y)

        if good.sum() < 5:
            continue

        r, p = stats.pearsonr(x[good], y[good])

        # simultaneous directional test:
        # cortex > 0 and hippocampal mobility < 0
        t_cx, p_cx = stats.ttest_1samp(x[good], 0.0)
        t_hc, p_hc = stats.ttest_1samp(y[good], 0.0)

        rows.append({
            "hemi": hemi,
            "n_subjects": int(good.sum()),
            "corr_cortex_hc_r": r,
            "corr_cortex_hc_p": p,
            "cortical_mean": np.mean(x[good]),
            "cortical_t": t_cx,
            "cortical_p": p_cx,
            "hipp_mobility_mean": np.mean(y[good]),
            "hipp_mobility_t": t_hc,
            "hipp_mobility_p": p_hc,
        })

    stats_df = pd.DataFrame(rows)
    stats_csv = OUT / "cortex_lowmode_vs_hipp_mobility_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    print(f"✅ wrote {stats_csv}")
    print(stats_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cortex-csv",
        default=None,
        help=(
            "Optional cortical per-run boundary beta CSV with columns "
            "sub, run, hemi, mode_k, beta."
        ),
    )

    args = ap.parse_args()

    boundary_locked_hc()

    if args.cortex_csv is not None:
        cortex_hc_coupling(Path(args.cortex_csv))


if __name__ == "__main__":
    main()