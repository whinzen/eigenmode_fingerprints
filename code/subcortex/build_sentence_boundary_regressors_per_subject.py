#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gamma

BASE = Path.home() / "eigenmode_fingerprints"
CSV_PATH = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"
PANG_OUT = BASE / "pang_out"

REG_PRE_DIR = PANG_OUT / "regressors" / "sentence_boundary_per_subject"
REG_HRF_DIR = PANG_OUT / "regressors" / "sentence_boundary_hrf_per_subject"
REG_PRE_DIR.mkdir(parents=True, exist_ok=True)
REG_HRF_DIR.mkdir(parents=True, exist_ok=True)

TR = 2.0


def canonical_hrf(tr, length=32.0):
    t = np.arange(0, length, tr)
    peak = gamma.pdf(t, 6)
    undershoot = gamma.pdf(t, 16)
    hrf = peak - 0.5 * undershoot
    if np.max(np.abs(hrf)) > 0:
        hrf = hrf / np.max(np.abs(hrf))
    return hrf


def get_subject_runs(sub_dir: Path):
    runs = []
    for rdir in sorted(sub_dir.glob("run-*")):
        if rdir.is_dir():
            try:
                runs.append(int(rdir.name.split("-")[1]))
            except Exception:
                pass
    return sorted(runs)


def main():
    df = pd.read_csv(CSV_PATH)

    required = ["run_id", "snt_id", "token_id", "onset", "offset"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {CSV_PATH}: {missing}")

    df = df.copy()
    df["run_id"] = df["run_id"].astype(int)
    df["snt_id"] = df["snt_id"].astype(int)
    df["token_id"] = df["token_id"].astype(int)
    df = df.sort_values(["run_id", "snt_id", "token_id"]).reset_index(drop=True)

    # First token of each sentence = sentence boundary
    df["boundary"] = (
        (df["snt_id"] != df["snt_id"].shift(1)) |
        (df["run_id"] != df["run_id"].shift(1))
    ).astype(int)
    df.loc[0, "boundary"] = 1

    hrf = canonical_hrf(TR)
    stim_pre = {}
    stim_hrf = {}

    for run_id, g in df.groupby("run_id", sort=True):
        g = g.sort_values(["snt_id", "token_id"]).reset_index(drop=True)
        onsets = g.loc[g["boundary"] == 1, "onset"].values.astype(float)

        T = int(np.ceil(g["offset"].max() / TR)) + 1
        x = np.zeros(T, dtype=float)

        for t in onsets:
            idx = int(round(t / TR))
            if 0 <= idx < T:
                x[idx] += 1.0

        x_hrf = np.convolve(x, hrf)[:len(x)]

        stim_pre[run_id] = x
        stim_hrf[run_id] = x_hrf

    print(f"✅ Built stimulus-space boundary regressors for runs: {sorted(stim_pre.keys())}")

    subjects = sorted([p for p in PANG_OUT.glob("sub-*") if p.is_dir()])
    stim_run_ids = sorted(stim_pre.keys())

    for sub_dir in subjects:
        sub = sub_dir.name
        subj_runs = get_subject_runs(sub_dir)

        n = min(len(subj_runs), len(stim_run_ids))
        if n == 0:
            print(f"[skip] {sub}: no runs found")
            continue

        for i in range(n):
            subj_run = subj_runs[i]
            stim_run = stim_run_ids[i]

            out_pre = REG_PRE_DIR / f"{sub}_run-{subj_run:02d}.npy"
            out_hrf = REG_HRF_DIR / f"{sub}_run-{subj_run:02d}.npy"

            np.save(out_pre, stim_pre[stim_run])
            np.save(out_hrf, stim_hrf[stim_run])

        print(f"✅ {sub}: wrote {n} pre-HRF + HRF boundary regressors")


if __name__ == "__main__":
    main()