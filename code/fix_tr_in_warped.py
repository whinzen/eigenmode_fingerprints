#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import re
import nibabel as nib
import numpy as np
import argparse

RUN_RE = re.compile(r"run-(\d+)")

def set_tr(warped_path: Path, src_path: Path):
    wimg = nib.load(str(warped_path))
    simg = nib.load(str(src_path))

    # TR from source header
    szooms = simg.header.get_zooms()
    if len(szooms) < 4 or float(szooms[3]) == 0.0:
        raise ValueError(f"No valid TR in source: {src_path}")
    TR = float(szooms[3])

    # Build a 4D header: keep warped affine, keep spatial zooms, set TR
    hdr = wimg.header.copy()
    sx, sy, sz = [float(z) for z in hdr.get_zooms()[:3]]
    hdr.set_zooms((sx, sy, sz, TR))

    # IMPORTANT: use float data
    data = wimg.get_fdata(dtype=np.float32)
    out = nib.Nifti1Image(data, wimg.affine, hdr)
    try:
        out.set_sform(wimg.get_sform(), wimg.header.get_sform_code())
        out.set_qform(wimg.get_qform(), wimg.header.get_qform_code())
    except Exception:
        pass

    nib.save(out, str(warped_path))
    print(f"[fixed] {warped_path.name}: TR={TR:.3f}s, dtype=float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-deriv", required=True, help="ds003643/derivatives")
    ap.add_argument("--warped", required=True, help="warped_fsavg")
    ap.add_argument("--subject", default=None, help="e.g., sub-EN057 (optional)")
    args = ap.parse_args()

    src_root = Path(args.src_deriv).resolve()
    war_root = Path(args.warped).resolve()

    subs = [args.subject] if args.subject else sorted(p.name for p in war_root.glob("sub-EN*") if p.is_dir())
    for sub in subs:
        wfunc = war_root/sub/"func"
        sfunc = src_root/sub/"func"
        if not wfunc.exists():
            print(f"[skip] no warped func for {sub}")
            continue
        for w in sorted(wfunc.glob("*_space-fsavgVol_*bold.nii.gz")):
            m = RUN_RE.search(w.name)
            if not m:
                print(f"[warn] cannot find run-XX in {w.name}")
                continue
            run = m.group(1)
            src = sorted(sfunc.glob(f"*run-{run}_space-MNIColin27_*bold.nii.gz"))
            if not src:
                print(f"[warn] no source for {sub} run-{run}")
                continue
            set_tr(w, src[0])

if __name__ == "__main__":
    main()