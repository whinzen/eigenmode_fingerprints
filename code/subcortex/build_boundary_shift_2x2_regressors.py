#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

BOUNDARY_DIR = PANG / "regressors" / "sentence_boundary_hrf_per_subject"
SHIFT_DIR = PANG / "regressors" / "shift_hrf_per_subject"

OUT = PANG / "regressors" / "boundary_shift_2x2_hrf_per_subject"

THRESH_Q_HIGH = 0.75
THRESH_Q_LOW = 0.25


def main():
    if OUT.exists():
        print(f"[note] output folder exists: {OUT}")
        print("[note] delete it first if you want a clean rebuild:")
        print(f"       rm -rf {OUT}")

    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(BOUNDARY_DIR.glob("sub-EN*_run-*.npy"))
    rows = []

    for bf in files:
        name = bf.name
        sf = SHIFT_DIR / name

        if not sf.exists():
            print(f"[skip] missing shift: {sf}")
            continue

        boundary = np.load(bf).squeeze().astype(float)
        shift = np.load(sf).squeeze().astype(float)

        n = min(len(boundary), len(shift))
        boundary = boundary[:n]
        shift = shift[:n]

        # These are HRF-convolved TR-level regressors.
        # We define approximate high/low boundary and high/low shift TRs
        # using within-run quantiles.
        b_hi = np.nanquantile(boundary, THRESH_Q_HIGH)
        b_lo = np.nanquantile(boundary, THRESH_Q_LOW)
        s_hi = np.nanquantile(shift, THRESH_Q_HIGH)
        s_lo = np.nanquantile(shift, THRESH_Q_LOW)

        boundary_high = boundary >= b_hi
        boundary_low = boundary <= b_lo

        shift_high = shift >= s_hi
        shift_low = shift <= s_lo

        masks = {
            "boundary_high_shift": boundary_high & shift_high,
            "boundary_low_shift": boundary_high & shift_low,
            "nonboundary_high_shift": boundary_low & shift_high,
            "nonboundary_low_shift": boundary_low & shift_low,
        }

        for reg_name, mask in masks.items():
            # Sparse binary condition regressor.
            # The downstream GLM script will z-score this vector.
            x = mask.astype(float)

            outdir = OUT / reg_name
            outdir.mkdir(parents=True, exist_ok=True)

            out = outdir / name
            np.save(out, x)

            rows.append({
                "file": name,
                "regressor": reg_name,
                "n_trs": n,
                "n_ones": int(mask.sum()),
                "fraction_ones": float(mask.mean()),
                "boundary_hi_thr": float(b_hi),
                "boundary_lo_thr": float(b_lo),
                "shift_hi_thr": float(s_hi),
                "shift_lo_thr": float(s_lo),
                "out_file": str(out),
            })

        print(f"✅ {name}")

    summary = pd.DataFrame(rows)
    summary_file = OUT / "boundary_shift_2x2_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"\n✅ wrote {summary_file}")
    print(summary.groupby("regressor")["fraction_ones"].describe())


if __name__ == "__main__":
    main()