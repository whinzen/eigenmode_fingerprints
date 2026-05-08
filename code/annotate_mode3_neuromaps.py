#!/usr/bin/env python

import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path

NEUROMAPS_DATA = Path.home() / "neuromaps-data"

from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from neuromaps.datasets import fetch_annotation
from neuromaps import transforms

BASE = Path.home() / "eigenmode_fingerprints"
EIG_DIR = BASE / "modes" / "fsaverage5"
OUT_DIR = BASE / "pang_out" / "mode3_annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODE_K = 3

L_MODE = EIG_DIR / "L_phi.npy"
R_MODE = EIG_DIR / "R_phi.npy"

# Add/edit after inspecting neuromaps_available_annotations.txt
ANNOTATIONS = [
    {
        "name": "Margulies_FC_gradient1",
        "source": "margulies2016",
        "desc": "fcgradient01",
        "space": "fsLR",
        "den": "32k",
    },
    {
        "name": "HCP_myelin_T1wT2w",
        "source": "hcps1200",
        "desc": "myelinmap",
        "space": "fsLR",
        "den": "32k",
    },
    {
        "name": "HCP_cortical_thickness",
        "source": "hcps1200",
        "desc": "thickness",
        "space": "fsLR",
        "den": "32k",
    },
]


def load_gifti_data(x):
    if isinstance(x, (str, Path)):
        img = nib.load(str(x))
    else:
        img = x

    if hasattr(img, "darrays"):
        if len(img.darrays) == 1:
            return np.asarray(img.darrays[0].data, float)
        return np.asarray(img.darrays[0].data, float)

    raise ValueError(f"Cannot read data from object: {type(x)}")


def fetch_and_transform(spec):
    print(f"\nFetching {spec['name']}")

    try:        
        annot = fetch_annotation(
    		source=spec["source"],
    		desc=spec["desc"],
    		space=spec["space"],
    		den=spec["den"],
		)
		
    except Exception as e:
        print(f"[skip] could not fetch {spec['name']}: {e}")
        return None

    space = spec["space"].lower()
    den = spec["den"].lower()

    try:
        if space == "fslr":
            annot_fsavg = transforms.fslr_to_fsaverage(
                annot,
                target_density="10k",
            )
        elif space == "fsaverage" and den != "10k":
            annot_fsavg = transforms.fsaverage_to_fsaverage(
                annot,
                target_density="10k",
            )
        elif space == "fsaverage" and den == "10k":
            annot_fsavg = annot
        elif space == "mni152":
            annot_fsavg = transforms.mni152_to_fsaverage(
                annot,
                target_density="10k",
            )
        else:
            print(f"[skip] unsupported space: {spec['space']}")
            return None
    except Exception as e:
        print(f"[skip] could not transform {spec['name']}: {e}")
        return None

    if not isinstance(annot_fsavg, (tuple, list)) or len(annot_fsavg) < 2:
        print(f"[skip] transformed annotation is not L/R surface pair")
        return None

    L = load_gifti_data(annot_fsavg[0])
    R = load_gifti_data(annot_fsavg[1])

    return L, R


def corr_one(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    good = np.isfinite(x) & np.isfinite(y)

    if good.sum() < 100:
        return {
            "n": good.sum(),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pr, pp = pearsonr(x[good], y[good])
    sr, sp = spearmanr(x[good], y[good])

    return {
        "n": int(good.sum()),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def main():
    phi_L = np.load(L_MODE)
    phi_R = np.load(R_MODE)

    mode_L = phi_L[:, MODE_K]
    mode_R = phi_R[:, MODE_K]

    print("Mode shapes:", mode_L.shape, mode_R.shape)

    rows = []

    for spec in ANNOTATIONS:
        data = fetch_and_transform(spec)

        if data is None:
            continue

        map_L, map_R = data

        if len(map_L) != len(mode_L) or len(map_R) != len(mode_R):
            print(
                f"[skip] vertex mismatch for {spec['name']}: "
                f"L {len(map_L)} R {len(map_R)}"
            )
            continue

        for hemi, mode, annot in [
            ("L", mode_L, map_L),
            ("R", mode_R, map_R),
        ]:
            res = corr_one(mode, annot)
            res.update({
                "annotation": spec["name"],
                "hemi": hemi,
                "mode_k": MODE_K,
                "abs_pearson_r": abs(res["pearson_r"]),
                "abs_spearman_rho": abs(res["spearman_rho"]),
            })
            rows.append(res)

        both_mode = np.concatenate([mode_L, mode_R])
        both_map = np.concatenate([map_L, map_R])
        res = corr_one(both_mode, both_map)
        res.update({
            "annotation": spec["name"],
            "hemi": "bihemi",
            "mode_k": MODE_K,
            "abs_pearson_r": abs(res["pearson_r"]),
            "abs_spearman_rho": abs(res["spearman_rho"]),
        })
        rows.append(res)

    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "mode3_neuromaps_correlations.csv"
    out.to_csv(out_csv, index=False)

    print(f"\n✅ wrote {out_csv}")
    print(out)


if __name__ == "__main__":
    main()