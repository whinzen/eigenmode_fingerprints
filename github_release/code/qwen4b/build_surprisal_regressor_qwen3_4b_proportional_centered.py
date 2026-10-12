#!/usr/bin/env python

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from build_transition_metric_regressors_qwen3_4b_proportional_centered import (
    BASE,
    FUNC_DIR,
    OUT_BASE,
    TR,
    HRF,
    get_n_trs_from_gifti,
    get_subject_run_mapping,
    find_left_func_file,
    build_impulse_regressor,
    convolve_hrf,
    zscore_safe,
)

SURP_CSV = (
    BASE
    / "pang_out"
    / "token_surprisal_qwen3_4b_proportional"
    / "token_surprisal_qwen3_4b_continuous.csv"
)

OUT_DIR = (
    OUT_BASE
    / "surprisal_sum_centered_hrf_per_subject"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    if not SURP_CSV.exists():
        raise FileNotFoundError(SURP_CSV)

    surp = pd.read_csv(SURP_CSV)

    required = {
        "run_id",
        "onset",
        "surprisal_sum",
    }

    missing = required.difference(surp.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    surp["run_id"] = surp["run_id"].astype(int)

    # ----------------------------------------------
    # Center native word surprisal within narrative run
    # ----------------------------------------------

    surp["surprisal_sum_centered"] = (
        surp["surprisal_sum"]
        - surp.groupby("run_id")["surprisal_sum"].transform("mean")
    )

    print("Input:", SURP_CSV)
    print("Rows:", len(surp))
    print(
        "Valid surprisal:",
        surp["surprisal_sum"].notna().sum(),
    )
    print(
        "Missing surprisal:",
        surp["surprisal_sum"].isna().sum(),
    )

    print("\nWithin-run centered means:")
    print(
        surp.groupby("run_id")[
            "surprisal_sum_centered"
        ].mean()
    )

    subjects = sorted([
        p.name
        for p in FUNC_DIR.glob("sub-*")
        if p.is_dir()
    ])

    print("\nSubjects:", len(subjects))
    print("Output:", OUT_DIR)

    n_saved = 0

    for subj in tqdm(subjects, desc="Subjects"):

        func_subj_dir = FUNC_DIR / subj / "func"

        if not func_subj_dir.exists():
            continue

        mapping = get_subject_run_mapping(
            func_subj_dir
        )

        if not mapping:
            continue

        for ann_run_id, actual_run in mapping.items():

            func_file = find_left_func_file(
                func_subj_dir,
                actual_run,
            )

            if func_file is None:
                continue

            n_trs = get_n_trs_from_gifti(
                func_file
            )

            run_df = surp[
                surp["run_id"] == ann_run_id
            ].copy()

            if run_df.empty:
                continue

            onsets = run_df["onset"].values
            vals = run_df[
                "surprisal_sum_centered"
            ].values

            good = (
                np.isfinite(onsets)
                & np.isfinite(vals)
            )

            if good.sum() == 0:

                x = np.zeros(
                    n_trs,
                    dtype=float,
                )

            else:

                x = build_impulse_regressor(
                    onsets[good],
                    vals[good],
                    n_trs=n_trs,
                    tr=TR,
                )

                x = convolve_hrf(
                    x,
                    HRF,
                )

                x = zscore_safe(x)

            out_path = (
                OUT_DIR
                / f"{subj}_run-{actual_run}.npy"
            )

            np.save(out_path, x)
            n_saved += 1

    print(
        f"\nDone. Saved {n_saved} surprisal regressors."
    )
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()
