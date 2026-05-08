#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import argparse
import re
import shutil

def make_clean_name(src_name: str) -> str:
    # Turn e.g. sub-EN057_task-lppEN_run-15_space-MNIColin27_desc-preproc_bold.nii.gz
    # into       sub-EN057_task-lppEN_run-15_space-fsavgVol_desc-preproc_bold.nii.gz
    if "_space-MNIColin27_" in src_name:
        return src_name.replace("_space-MNIColin27_", "_space-fsavgVol_")
    # fallback: append suffix before .nii.gz
    return src_name.replace(".nii.gz", "_space-fsavgVol.nii.gz")

def rename_subject(src_deriv_sub: Path, warped_sub: Path):
    src_files = sorted(src_deriv_sub.glob("*task-lppEN*desc-preproc_bold.nii.gz"))
    war_files = sorted(warped_sub.glob("*.nii.gz"))  # currently MD5E-... names

    if not src_files or not war_files:
        print(f"[skip] no files: src={len(src_files)} warped={len(war_files)} in {warped_sub}")
        return

    if len(src_files) != len(war_files):
        print(f"[warn] count mismatch in {warped_sub}: src={len(src_files)} vs warped={len(war_files)}")

    # Map in lexicographic order (derivatives are BIDS-sorted: run-01 .. run-09)
    for src, w in zip(src_files, war_files):
        new_name = make_clean_name(src.name)
        target = warped_sub / new_name
        if target.exists():
            print(f"[skip] exists: {target.name}")
            continue
        print(f"[rename] {w.name}  ->  {new_name}")
        w.replace(target)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-deriv", required=True, help="Path to ds003643/derivatives")
    ap.add_argument("--warped", required=True, help="Path to warped_fsavg (root)")
    ap.add_argument("--subject", default=None, help="Optional: single subject, e.g., sub-EN057")
    args = ap.parse_args()

    src_root = Path(args.src_deriv).resolve()
    warped_root = Path(args.warped).resolve()

    if args.subject:
        subs = [args.subject]
    else:
        subs = sorted([p.name for p in warped_root.glob("sub-EN*") if p.is_dir()])

    for sub in subs:
        src_sub = src_root / sub / "func"
        war_sub = warped_root / sub / "func"
        if not war_sub.exists():
            print(f"[skip] no warped func dir for {sub}")
            continue
        rename_subject(src_sub, war_sub)

if __name__ == "__main__":
    main()