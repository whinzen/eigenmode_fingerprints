#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from nilearn.glm.first_level import spm_hrf

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

# PRE-HRF regressors
BOUNDARY_DIR = PANG / "regressors" / "sentence_boundary_per_subject"
SHIFT_DIR = PANG / "regressors" / "shift_per_subject"

OUT = PANG / "regressors" / "boundary_token_shift_2x2_prehrf_then_hrf"
OUT.mkdir(parents=True, exist_ok=True)

TR = 2.0

HIGH_Q = 0.75
LOW_Q = 0.25


def hrf_convolve(x, tr=2.0):
    hrf = spm_hrf(t_r=tr)
    return np.convolve(x, hrf)[: len(x)]


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

    # boundary events
    boundary_event = boundary.astype(bool)

    # finite shift values
    finite = np.isfinite(shift)

    shift_clean = np.zeros_like(shift)
    shift_clean[finite] = shift[finite]

    # define quantiles ONLY from actual shift events
    event_vals = shift_clean[shift_clean > 0]

    if len(event_vals) == 0:
        print(f"[skip] no nonzero shift events for {name}")
        return []

    hi_thr = np.quantile(event_vals, HIGH_Q)
    lo_thr = np.quantile(event_vals, LOW_Q)

    high_shift = shift_clean >= hi_thr
    low_shift = (shift_clean > 0) & (shift_clean <= lo_thr)

    masks = {
        "boundary_high_shift":
            boundary_event & high_shift,

        "boundary_low_shift":
            boundary_event & low_shift,

        "nonboundary_high_shift":
            (~boundary_event) & high_shift,

        "nonboundary_low_shift":
            (~boundary_event) & low_shift,
    }

    rows = []

    for cond, mask in masks.items():

        # binary event regressor
        x_binary_pre = mask.astype(float)

        # weighted regressor
        x_weighted_pre = np.zeros_like(shift_clean)
        x_weighted_pre[mask] = shift_clean[mask]

        for kind, x_pre in [
            ("binary", x_binary_pre),
            ("weighted", x_weighted_pre),
        ]:

            x_hrf = hrf_convolve(x_pre, TR)

            outdir = OUT / kind / cond
            outdir.mkdir(parents=True, exist_ok=True)

            outfile = outdir / name
            np.save(outfile, x_hrf)

            rows.append({
                "file": name,
                "kind": kind,
                "condition": cond,
                "n_trs": len(x_pre),
                "n_events_prehrf": int(mask.sum()),
                "fraction_events_prehrf": float(mask.mean()),
                "shift_hi_thr": float(hi_thr),
                "shift_lo_thr": float(lo_thr),
                "outfile": str(outfile),
            })

    print(f"✅ {name}")

    return rows


def main():

    files = sorted(BOUNDARY_DIR.glob("sub-EN*_run-*.npy"))

    if len(files) == 0:
        raise RuntimeError(
            f"No files found in {BOUNDARY_DIR}"
        )

    all_rows = []

    for bf in files:
        rows = build_for_file(bf.name)
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)

    summary_file = (
        OUT / "boundary_token_shift_2x2_summary.csv"
    )

    summary.to_csv(summary_file, index=False)

    print(f"\n✅ wrote {summary_file}")

    print(
        summary.groupby(
            ["kind", "condition"]
        )["fraction_events_prehrf"].describe()
    )


if __name__ == "__main__":
    main()