#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

REGRESSOR_ROOT = PANG / "regressors"
OUT_ROOT = PANG / "regressors" / "circular_shift_controls"

SOURCE_DIRS = {
    "sentence_shift": REGRESSOR_ROOT / "sentence_shift_hrf_per_subject",
    "boundary": REGRESSOR_ROOT / "sentence_boundary_hrf_per_subject",
    "token_shift": REGRESSOR_ROOT / "shift_hrf_per_subject",
    "pred_error_ar": REGRESSOR_ROOT / "pred_error_ar_hrf_per_subject",
    "pred_error_subspace": REGRESSOR_ROOT / "pred_error_subspace_hrf_per_subject",
    "curvature": REGRESSOR_ROOT / "curvature_hrf_per_subject",
}


def zscore_like_original(x):
    """
    Preserve the original scale? For control regressors, we simply circularly
    shift the original values and do not z-score here. The GLM script z-scores.
    """
    return np.asarray(x, float)


def circular_shift(x, min_shift=60, max_shift=None, rng=None):
    """
    Circularly shift a 1D regressor by a random amount.

    min_shift:
        minimum absolute TR shift. Should exceed the HRF width.
    max_shift:
        maximum shift. Defaults to n - min_shift.
    """

    x = np.asarray(x, float)
    n = len(x)

    if max_shift is None:
        max_shift = n - min_shift

    if n <= 2 * min_shift:
        # fallback: half-run shift for short runs
        shift = n // 2
    else:
        shift = rng.integers(min_shift, max_shift + 1)

    return np.roll(x, shift), int(shift)


def build_controls(metric, n_per_run=1, min_shift=60, seed=1234):
    if metric not in SOURCE_DIRS:
        raise ValueError(f"Unknown metric: {metric}")

    src_dir = SOURCE_DIRS[metric]

    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    rng = np.random.default_rng(seed)

    files = sorted(src_dir.glob("sub-EN*_run-*.npy"))

    if not files:
        raise RuntimeError(f"No .npy files found in {src_dir}")

    rows = []

    for f in files:
        x = np.load(f).squeeze().astype(float)

        for i in range(n_per_run):
            y, shift = circular_shift(
                x,
                min_shift=min_shift,
                rng=rng,
            )

            out_dir = OUT_ROOT / metric / f"shift-{i:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / f.name
            np.save(out_file, y)

            rows.append({
                "metric": metric,
                "source_file": str(f),
                "out_file": str(out_file),
                "filename": f.name,
                "n_trs": len(x),
                "shift_index": i,
                "shift_trs": shift,
                "seed": seed,
                "min_shift": min_shift,
            })

        print(f"✅ {metric}: {f.name}")

    summary = pd.DataFrame(rows)
    summary_file = OUT_ROOT / metric / "circular_shift_summary.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)

    print(f"\n✅ wrote {summary_file}")
    print(summary.head())
    print(summary["shift_trs"].describe())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metric",
        choices=list(SOURCE_DIRS.keys()) + ["all"],
        default="sentence_shift",
        help="Which regressor to circularly shift.",
    )
    ap.add_argument(
        "--n-per-run",
        type=int,
        default=1,
        help="Number of independent circular-shift controls per run.",
    )
    ap.add_argument(
        "--min-shift",
        type=int,
        default=60,
        help="Minimum circular shift in TRs.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )

    args = ap.parse_args()

    metrics = list(SOURCE_DIRS.keys()) if args.metric == "all" else [args.metric]

    for metric in metrics:
        build_controls(
            metric=metric,
            n_per_run=args.n_per_run,
            min_shift=args.min_shift,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()