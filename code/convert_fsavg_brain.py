#!/usr/bin/env python
import nibabel as nib
from pathlib import Path
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--fsaverage_mgz", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
img = nib.load(args.fsaverage_mgz)
nib.save(img, str(out))
print("Wrote:", out)
