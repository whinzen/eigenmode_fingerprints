#!/usr/bin/env python
from __future__ import annotations
import ants, nibabel as nib, numpy as np, os
from pathlib import Path
import argparse

# Limit ANTs threads to reduce RAM pressure
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")

def mean_bold_nifti(bold_path: Path, out_path: Path) -> Path:
    """Compute mean BOLD image (3D) from 4D NIfTI."""
    img = nib.load(str(bold_path))
    data = img.get_fdata()
    mean3d = np.mean(data, axis=3)
    mean_img = nib.Nifti1Image(mean3d, img.affine, img.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(mean_img, str(out_path))
    return out_path

def warp_run_to_fsavg(
    bold_path: Path,
    fsavg_brain: Path,
    out_dir: Path,
    reg_cache_dir: Path,
    transform: str = "SyN",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_cache_dir.mkdir(parents=True, exist_ok=True)
    bold_path = bold_path.resolve()
    bname = bold_path.name

    # Construct output filename with clean naming
    if "_space-MNIColin27_" in bname:
        out_bold = out_dir / bname.replace("_space-MNIColin27_", "_space-fsavgVol_")
    else:
        out_bold = out_dir / bname.replace(".nii.gz", "_space-fsavgVol.nii.gz")

    if out_bold.exists():
        print(f"[skip] already warped: {out_bold.name}")
        return out_bold

    # run-specific mean for registration
    mean_path = reg_cache_dir / (bname.replace(".nii.gz", "_mean.nii.gz"))
    mean_bold_nifti(bold_path, mean_path)

    fixed = ants.image_read(str(fsavg_brain))   # fsaverage brain
    moving = ants.image_read(str(mean_path))    # mean EPI

    print(f"[reg] {bname} → fsavg_brain [{transform}]")
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform)

    # Low-memory fallback: warp each volume separately
    print(f"[apply] warping 4D by volumes: {bname}")
    mov4d = ants.image_read(str(bold_path))
    mov_np = mov4d.numpy()  # shape (X, Y, Z, T)
    T = mov_np.shape[3]
    out_np = np.zeros(fixed.shape + (T,), dtype=np.float32)

    for t in range(T):
        if t % 50 == 0:
            print(f"  - volume {t+1}/{T}")
        vol_np = mov_np[..., t].astype(np.float32)

        # Build a 3D ANTs image and set geometry
        vol = ants.from_numpy(vol_np)
        try:
            sp = tuple(mov4d.spacing[:3])
            og = tuple(mov4d.origin[:3])
            d = np.array(mov4d.direction)
            if d.size >= 9:
                dim = int(round(np.sqrt(d.size)))  # 3 or 4
                d3 = d.reshape(dim, dim)[:3, :3]
                di = tuple(d3.ravel())
            else:
                di = (1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0)
            vol.set_spacing(sp)
            vol.set_origin(og)
            vol.set_direction(di)
        except Exception:
            # Fall back to fsaverage geometry
            vol.set_spacing(fixed.spacing)
            vol.set_origin(fixed.origin)
            vol.set_direction(fixed.direction)

        wvol = ants.apply_transforms(
            fixed=fixed,
            moving=vol,
            transformlist=reg["fwdtransforms"],
            interpolator="bSpline",
        )
        out_np[..., t] = wvol.numpy().astype(np.float32)
    
        # Save as standard 4D NIfTI on fsaverage grid, with correct TR
    ref_img = nib.load(str(fsavg_brain))            # 3D fsaverage brain (for affine + spatial zooms)
    src_img = nib.load(str(bold_path))              # original 4D run (for 4D header + TR)

    # TR from source
    src_zooms = src_img.header.get_zooms()
    TR = float(src_zooms[3]) if len(src_zooms) >= 4 else 1.0

    # Build a 4D header by starting from the source header (which is 4D),
    # then overwrite the first 3 zooms with fsaverage spatial zooms, keep the TR.
    src_hdr = src_img.header.copy()
    spatial_zooms = ref_img.header.get_zooms()[:3]  # dx, dy, dz from fsavg
    src_hdr.set_zooms(spatial_zooms + (TR,))        # dx,dy,dz,TR for 4D

    # Create the 4D image on fsaverage affine, with the 4D header
    out_img = nib.Nifti1Image(out_np, ref_img.affine, src_hdr)

    # Preserve sform/qform codes, if available, using fsaverage geometry
    try:
        out_img.set_sform(ref_img.get_sform(), ref_img.header.get_sform_code())
        out_img.set_qform(ref_img.get_qform(), ref_img.header.get_qform_code())
    except Exception:
        pass

    nib.save(out_img, str(out_bold))
    print(f"[ok] wrote: {out_bold} (TR={TR:.3f}s, T={out_np.shape[3]})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deriv", required=True, help="Path to derivatives/")
    ap.add_argument("--out", required=True, help="Output directory for warped runs")
    ap.add_argument("--fsavg_brain", required=True, help="fsaverage brain NIfTI")
    ap.add_argument(
        "--pattern",
        default="*task-lppEN*space-MNIColin27*desc-preproc_bold.nii.gz",
        help="Glob pattern to match BOLD runs",
    )
    ap.add_argument("--cache", default="reg_cache", help="Cache directory for mean EPIs")
    ap.add_argument(
        "--transform", default="SyN", help="Registration type: SyN (accurate) or Affine (fast)"
    )
    args = ap.parse_args()

    deriv = Path(args.deriv).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    fsavg_brain = Path(args.fsavg_brain).expanduser().resolve()
    cache = Path(args.cache).expanduser().resolve()

    runs = sorted(deriv.rglob(args.pattern))
    if not runs:
        print("No runs found under", deriv)
        return

    print(f"Found {len(runs)} runs.")
    for rp in runs:
        try:
            subj_out = out_root / rp.parent.relative_to(deriv)
            reg_cache_dir = cache / rp.parent.relative_to(deriv)
            warp_run_to_fsavg(rp, fsavg_brain, subj_out, reg_cache_dir, transform=args.transform)
        except Exception as e:
            print("[error]", rp, "::", e)

if __name__ == "__main__":
    main()