#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

BASE = Path.home() / "eigenmode_fingerprints"
DATA_ROOT = BASE / "data" / "empirical"

MASK_DIR = BASE / "pang_out" / "subcortex" / "brainstem_masks_resampled"
OUT = BASE / "pang_out" / "subcortex" / "brainstem_roi_timeseries"
OUT.mkdir(parents=True, exist_ok=True)

ROIS = {
    "LC_L":  ("LC_L_BrainstemNavigator_prob0.35_resampled_to_LPP_bold.nii.gz", 0.10),
    "LC_R":  ("LC_R_BrainstemNavigator_prob0.35_resampled_to_LPP_bold.nii.gz", 0.10),
    "VTA_L": ("VTA_L_BrainstemNavigator_prob0.35_resampled_to_LPP_bold.nii.gz", 0.25),
    "VTA_R": ("VTA_R_BrainstemNavigator_prob0.35_resampled_to_LPP_bold.nii.gz", 0.25),
}


def infer_run(path):
    for p in path.name.split("_"):
        if p.startswith("run-"):
            return p.replace("run-", "")
    raise ValueError(path)


def main():
    masks = {}

    for roi, (fname, frac) in ROIS.items():
        f = MASK_DIR / fname
        data = nib.load(str(f)).get_fdata()
        thr = frac * np.nanmax(data)
        mask = data > thr
        masks[roi] = mask
        print(f"{roi}: threshold={thr:.6f}, voxels={mask.sum()}")

    bold_files = sorted(
        DATA_ROOT.glob(
            "sub-EN*/func/*task-lppEN*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
    )

    rows = []

    for bold_file in bold_files:
        sub = bold_file.parts[-3]
        run = f"{int(infer_run(bold_file)):02d}"

        bold = nib.load(str(bold_file)).get_fdata()

        for roi, mask in masks.items():
            if mask.shape != bold.shape[:3]:
                raise ValueError(f"Shape mismatch for {roi}: {mask.shape} vs {bold.shape[:3]}")

            ts = np.nanmean(bold[mask, :], axis=0)

            out_file = OUT / f"{sub}_run-{run}_{roi}_mean_signal.npy"
            np.save(out_file, ts)

            rows.append({
                "subject": sub,
                "run": run,
                "roi": roi,
                "n_voxels": int(mask.sum()),
                "n_trs": len(ts),
                "timeseries_file": str(out_file),
                "bold_file": str(bold_file),
            })

        print(f"✅ {sub} run-{run}")

    index = pd.DataFrame(rows)
    out_csv = OUT / "brainstem_roi_timeseries_index.csv"
    index.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(index.head())


if __name__ == "__main__":
    main()