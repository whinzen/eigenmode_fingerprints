
# lpp_batch_runner.py (advanced, EN-only, with per-subject run→section auto-mapping)
# -------------------------------------------------------
from __future__ import annotations
import argparse, json, re, sys, os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from . import lpp_eigenmode_pipeline_textgrid as lpp


def find_english_runs(derivatives_dir: Path):
    """Return list of (subject_id, bold_path, run_no) for English runs."""
    rows = []
    for bold_path in derivatives_dir.rglob("*_desc-preproc_bold.nii.gz"):
        name = bold_path.name
        if "task-lppEN" not in name:
            continue
        m = re.search(r"(sub-[^/]+)", str(bold_path))
        sub = m.group(1) if m else "unknown"
        r = re.search(r"run-(\d+)", name)
        run_no = int(r.group(1)) if r else None
        rows.append((sub, bold_path, run_no))
    # sort per subject by run number
    rows.sort(key=lambda x: (x[0], x[2] if x[2] is not None else 1_000_000))
    return rows


def build_sequential_mapping(subject_runs):
    """
    subject_runs: list of tuples (sub, bold_path, run_no) for ONE subject, sorted by run_no.
    Returns dict: {run_no: [section_index]} mapping 1..9 across the available runs.
    """
    mapping = {}
    sec = 1
    for _, _, run_no in subject_runs:
        if run_no is None:
            continue
        if sec > 9:
            break
        mapping[run_no] = [sec]
        sec += 1
    return mapping


def compute_summary_metrics(lam, E):
    lam = np.asarray(lam); E = np.asarray(E)
    good = np.isfinite(lam) & (lam > 0) & np.isfinite(E) & (E > 0)
    lam = lam[good]; E = E[good]

    # need enough points to compute anything stable
    if lam.size < 10:
        return np.nan, np.nan, np.nan

    f = np.sqrt(lam)
    p = E / E.sum()
    PR = (E.sum() ** 2) / (E @ E)
    SE = -(p * np.log(p)).sum()

    # robust middle slice for slope
    K = len(E)
    i0, i1 = int(0.1*K), int(0.9*K)
    if i1 <= i0:
        return PR, SE, np.nan
    x = np.log(f[i0:i1]); y = np.log(E[i0:i1])
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if x.size < 5:
        return PR, SE, np.nan

    try:
        beta = np.polyfit(x, y, 1)[0]
    except Exception:
        beta = np.nan
    return PR, SE, beta


def textgrid_length_seconds(tg_path: Path) -> float:
    txt = Path(tg_path).read_text(encoding="utf-8", errors="ignore")
    ms = list(re.finditer(r"xmax\s*=\s*([0-9.]+)", txt))
    return float(ms[-1].group(1)) if ms else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deriv", required=True, help="Path to derivatives directory")
    ap.add_argument("--stim", required=True, help="Path to directory with lppEN_section*.TextGrid (e.g., annotation/EN)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--fsaverage", default="fsaverage5")
    ap.add_argument("--kmodes", type=int, default=200)
    ap.add_argument("--map", default=None, help="Optional JSON mapping override: { 'sub-EN001': { 'run-07': [1], ... } }")
    args = ap.parse_args()

    derivatives = Path(args.deriv).expanduser().resolve()
    stim_dir = Path(args.stim).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Optional: external mapping override
    override = None
    if args.map:
        with open(args.map, "r") as f:
            override = json.load(f)

    rows = find_english_runs(derivatives)
    if not rows:
        print("No English runs (task-lppEN) found under", derivatives)
        sys.exit(1)

    # index by subject
    by_sub = {}
    for sub, bold_path, run_no in rows:
        by_sub.setdefault(sub, []).append((sub, bold_path, run_no))

    # Pre-index available EN textgrids
    tg_index = sorted(glob.glob(str(stim_dir / "lppEN_section*.TextGrid")))

    summary_rows = []

    for sub, subject_runs in by_sub.items():
        # subject-specific mapping: override JSON > sequential
        if override and sub in override:
            run_to_sections = {int(k.split("-")[1]): v for k, v in override[sub].items()}
            mapping_src = "OVERRIDE"
        else:
            subject_runs.sort(key=lambda x: (x[2] if x[2] is not None else 1_000_000))
            run_to_sections = build_sequential_mapping(subject_runs)
            mapping_src = "SEQUENTIAL-9"

        # print mapping
        human_map = ", ".join([f"run-{r:02d}→sec{s[0]}" for r, s in sorted(run_to_sections.items())])
        print(f"\n=== {sub} mapping ({mapping_src}) ===")
        print(human_map if human_map else "(No runs mapped)")

        # process each run that appears in the mapping
        for _, bold_path, run_no in subject_runs:
            if run_no not in run_to_sections:
                continue

            sec_idx = run_to_sections[run_no][0]
            tg_path = stim_dir / f"lppEN_section{sec_idx}.TextGrid"
            if not tg_path.exists():
                print(f"Missing TextGrid file: {tg_path.name} — skipping.")
                continue

            print(f"\n--- {sub} :: {bold_path.name} (mapped to section {sec_idx}) ---")

            # Sanity: run duration vs. TextGrid
            img = nib.load(str(bold_path))
            tr  = img.header.get_zooms()[3]
            nT  = img.shape[3]
            dur_run = tr * nT

            dur_tg = textgrid_length_seconds(tg_path)
            ratio = dur_run / max(dur_tg, 1e-6)
            status = "1:1 OK" if 0.8 <= ratio <= 1.2 else (">1 section?" if ratio > 1.2 else "<1 section?")
            print(f"Sanity: run {dur_run:.1f}s vs TG(sec{sec_idx}) {dur_tg:.1f}s (ratio {ratio:.2f}) -> {status}")

            # Output dir
            run_dir = out_root / sub / bold_path.name.replace("_desc-preproc_bold.nii.gz", "")
            run_dir.mkdir(parents=True, exist_ok=True)

            # Run pipeline with an explicit per-run mapping
            out = lpp.run_lpp_textgrid_pipeline(
                bold_path=bold_path,
                stim_dir=stim_dir,
                lang="EN",
                fs_level=args.fsaverage,
                K_modes=args.kmodes,
                bands=None,
                run_to_sections={run_no: run_to_sections[run_no]},
            )

            lam = np.asarray(out["lam"])
            E   = np.asarray(out["E_mean"])
            valid = np.isfinite(lam) & (lam > 0) & np.isfinite(E) & (E > 0)
            lam, E = lam[valid], E[valid]
            if lam.size == 0:
                print("No valid modes after filtering; skipping this run.")
                continue

            # Save spectrum CSV
            pd.DataFrame({
                "k": np.arange(1, len(lam) + 1),
                "lambda": lam,
                "sqrt_lambda": np.sqrt(lam),
                "E_k": E
            }).to_csv(run_dir / "spectrum.csv", index=False)

            # Save ERME CSVs
            for name, y in out["ERME"].items():
                pd.DataFrame({"lag_s": out["lags"], "delta_energy": y}).to_csv(run_dir / f"erme_{name}.csv", index=False)

            # Plots
            plt.figure(figsize=(6, 4))
            plt.loglog(np.sqrt(lam), E, marker='o', linewidth=1)
            plt.xlabel("Spatial frequency ~ sqrt(lambda)")
            plt.ylabel("Mode energy (variance)")
            plt.title(f"{sub} :: {bold_path.name} :: Spectrum")
            plt.grid(True); plt.tight_layout()
            plt.savefig(run_dir / "spectrum.png", dpi=150); plt.close()

            plt.figure(figsize=(6, 4))
            for name, y in out["ERME"].items():
                plt.plot(out["lags"], y, label=name)
            plt.xlabel("Lag (s) from sentence boundary")
            plt.ylabel("Δ Mode energy (band-avg)")
            plt.title(f"{sub} :: {bold_path.name} :: ERME")
            plt.axvline(0, linestyle='--')
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(run_dir / "erme.png", dpi=150); plt.close()

            # Summary metrics
            PR, SE, beta = compute_summary_metrics(lam, E)
            summary_rows.append({
                "subject": sub,
                "run": bold_path.name,
                "mapped_section": sec_idx,
                "PR": PR,
                "spectral_entropy": SE,
                "beta": beta,
                "TR": out["TR"],
                "n_modes": len(lam)
            })

    pd.DataFrame(summary_rows).to_csv(out_root / "summary_per_run.csv", index=False)
    print("\nWrote summary to", out_root / "summary_per_run.csv")


if __name__ == "__main__":
    main()