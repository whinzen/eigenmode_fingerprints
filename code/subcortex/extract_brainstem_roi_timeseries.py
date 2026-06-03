#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_dilation


BASE = Path.home() / "eigenmode_fingerprints"
DATA_ROOT = BASE / "data" / "empirical"
OUT = BASE / "pang_out" / "subcortex" / "brainstem_roi_timeseries"
OUT.mkdir(parents=True, exist_ok=True)


def infer_run_id(path):
    for part in path.name.split("_"):
        if part.startswith("run-"):
            return part.replace("run-", "")
    raise ValueError(f"Could not infer run: {path}")


def load_mask(mask_file, bold_img, threshold=0.0, dilate=0):
    mask_img = nib.load(str(mask_file))
    mask = mask_img.get_fdata()

    if mask.shape != bold_img.shape[:3]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match BOLD shape {bold_img.shape[:3]}. "
            "Mask must already be in same MNI grid as functional data."
        )

    mask = mask > threshold

    if dilate > 0:
        for _ in range(dilate):
            mask = binary_dilation(mask)

    return mask


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--vta-mask", required=True)
    ap.add_argument("--lc-mask", required=True)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--dilate", type=int, default=0)

    args = ap.parse_args()

    masks = {
        "VTA": Path(args.vta_mask),
        "LC": Path(args.lc_mask),
    }

    bold_files = sorted(
        DATA_ROOT.glob(
            "sub-EN*/func/*task-lppEN*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
    )

    rows = []

    print(f"Found BOLD files: {len(bold_files)}")

    for bold_file in bold_files:
        sub = bold_file.parts[-3]
        run = infer_run_id(bold_file)
        run_padded = f"{int(run):02d}"

        bold_img = nib.load(str(bold_file))
        bold = bold_img.get_fdata()

        for roi, mask_file in masks.items():
            try:
                mask = load_mask(
                    mask_file=mask_file,
                    bold_img=bold_img,
                    threshold=args.threshold,
                    dilate=args.dilate,
                )
            except Exception as e:
                print(f"[skip] {sub} run-{run_padded} {roi}: {e}")
                continue

            n_voxels = int(mask.sum())

            if n_voxels < 3:
                print(f"[skip] {sub} run-{run_padded} {roi}: only {n_voxels} voxels")
                continue

            X = bold[mask, :]
            ts = np.nanmean(X, axis=0)

            out_file = OUT / f"{sub}_run-{run_padded}_{roi}_mean_signal.npy"
            np.save(out_file, ts)

            rows.append({
                "subject": sub,
                "run": run_padded,
                "roi": roi,
                "n_voxels": n_voxels,
                "n_trs": len(ts),
                "timeseries_file": str(out_file),
                "bold_file": str(bold_file),
                "mask_file": str(mask_file),
            })

            print(f"✅ {sub} run-{run_padded} {roi}: {n_voxels} voxels")

    index = pd.DataFrame(rows)
    index_file = OUT / "brainstem_roi_timeseries_index.csv"
    index.to_csv(index_file, index=False)

    print(f"\nWrote {index_file}")
    print(index.head())


if __name__ == "__main__":
    main()