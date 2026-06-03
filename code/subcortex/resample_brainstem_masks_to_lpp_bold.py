#!/usr/bin/env python

from pathlib import Path
import nibabel as nib
from nilearn.image import resample_to_img

BASE = Path.home()

ATLAS = (
    Path.home()
    / "eigenmode_fingerprints"
    / "BrainstemNavigatorv1.0"
    / "1.0"
    / "2a.BrainstemNucleiAtlas_MNI"
)
MASK_DIR = ATLAS / "labels_thresholded_probabilistic_0.35"

REF_BOLD = (
    BASE / "eigenmode_fingerprints" / "data" / "empirical" /
    "sub-EN058" / "func" /
    "sub-EN058_task-lppEN_run-10_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)

OUT = (
    BASE / "eigenmode_fingerprints" /
    "pang_out" / "subcortex" / "brainstem_masks_resampled"
)
OUT.mkdir(parents=True, exist_ok=True)

MASKS = {
    "LC_L": MASK_DIR / "LC_l.nii.gz",
    "LC_R": MASK_DIR / "LC_r.nii.gz",
    "VTA_L": MASK_DIR / "VTA_PBP_l.nii.gz",
    "VTA_R": MASK_DIR / "VTA_PBP_r.nii.gz",
}


def main():
    ref = nib.load(str(REF_BOLD))

    for name, mask_file in MASKS.items():
        print(f"Resampling {name}: {mask_file}")

        img = nib.load(str(mask_file))

        res = resample_to_img(
            source_img=img,
            target_img=ref,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )

        out_file = OUT / f"{name}_BrainstemNavigator_prob0.35_resampled_to_LPP_bold.nii.gz"
        nib.save(res, str(out_file))

        data = res.get_fdata()
        n_vox = int((data > 0).sum())

        print(f"  wrote {out_file}")
        print(f"  voxels > 0: {n_vox}")

    print(f"\nDone. Output directory:\n{OUT}")


if __name__ == "__main__":
    main()