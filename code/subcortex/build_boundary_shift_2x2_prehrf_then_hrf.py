#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from nilearn.glm.first_level import spm_hrf

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

# These should be PRE-HRF / unconvolved TR-binned regressors.
BOUNDARY_DIR = PANG / "regressors" / "sentence_boundary_per_subject"
SHIFT_DIR = PANG / "regressors" / "sentence_shift_per_subject"

OUT = PANG / "regressors" / "boundary_shift_2x2_prehrf_then_hrf"
OUT.mkdir(parents=True, exist_ok=True)

TR = 2.0
HIGH_Q = 0.75
LOW_Q = 0.25


def hrf_convolve(x, tr=2.0):
    hrf = spm_hrf(t_r=tr)
    y = np.convolve(x, hrf)[: len(x)]
    return y


def load_pair(name):
    bf = BOUNDARY_DIR / name
    sf = SHIFT_DIR / name

    if not bf.exists() or not sf.exists():
        return None, None

    boundary = np.load(bf).squeeze().astype(float)
    shift = np.load(sf).squeeze().astype(float)

    n = min(len(boundary), len(shift))
    return boundary[:n], shift[:n]


def build_for_file(name):
    boundary, shift = load_pair(name)

    if boundary is None:
        print(f"[skip] missing pair for {name}")
        return []

    # Work PRE-HRF here.
    boundary_event = boundary.astype(bool)

    # Use only finite shift values.
    finite = np.isfinite(shift)

    shift_clean = np.zeros_like(shift)
    shift_clean[finite] = shift[finite]

    # only nonzero shift events define quantiles
    event_vals = shift_clean[shift_clean > 0]

    if len(event_vals) == 0:
        print(f"[skip] no nonzero shift events for {name}")
        return []

    hi_thr = np.quantile(event_vals, HIGH_Q)
    lo_thr = np.quantile(event_vals, LOW_Q)

    high_shift = shift_clean >= hi_thr
    low_shift = shift_clean <= lo_thr

    masks = {
        "boundary_high_shift": boundary_event & high_shift,
        "boundary_low_shift": boundary_event & low_shift,
        "nonboundary_high_shift": (~boundary_event) & high_shift,
        "nonboundary_low_shift": (~boundary_event) & low_shift,
    }

    rows = []

    for cond, mask in masks.items():
        # A. Binary event version
        x_binary_pre = mask.astype(float)

        # B. Magnitude-weighted version
        x_weighted_pre = np.zeros_like(shift_clean, dtype=float)
        x_weighted_pre[mask] = shift_clean[mask]

        for kind, x_pre in [
            ("binary", x_binary_pre),
            ("weighted", x_weighted_pre),
        ]:
            x_hrf = hrf_convolve(x_pre, TR)

            outdir = OUT / kind / cond
            outdir.mkdir(parents=True, exist_ok=True)

            out = outdir / name
            np.save(out, x_hrf)

            rows.append({
                "file": name,
                "kind": kind,
                "condition": cond,
                "n_trs": len(x_pre),
                "n_events_prehrf": int(mask.sum()),
                "fraction_events_prehrf": float(mask.mean()),
                "shift_hi_thr": float(hi_thr),
                "shift_lo_thr": float(lo_thr),
                "out_file": str(out),
            })

    print(f"✅ {name}")
    return rows


def main():
    files = sorted(BOUNDARY_DIR.glob("sub-EN*_run-*.npy"))

    if len(files) == 0:
        raise RuntimeError(
            f"No pre-HRF files found in {BOUNDARY_DIR}\n"
            "You need unconvolved/TR-binned boundary regressors first. "
            "Do not use sentence_boundary_hrf_per_subject here."
        )

    all_rows = []

    for bf in files:
        all_rows.extend(build_for_file(bf.name))

    summary = pd.DataFrame(all_rows)
    summary_file = OUT / "boundary_shift_2x2_prehrf_then_hrf_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"\n✅ wrote {summary_file}")
    print(summary.groupby(["kind", "condition"])["fraction_events_prehrf"].describe())


if __name__ == "__main__":
    main()