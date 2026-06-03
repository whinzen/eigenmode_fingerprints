#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN = PANG / "subcortex" / "vta_hipp_coupling" / "vta_hipp_timeseries_coupling_by_run.csv"
BOUNDARY_DIR = PANG / "regressors" / "sentence_boundary_per_subject"

OUT = PANG / "subcortex" / "vta_hipp_boundary_modulated_coupling"
OUT.mkdir(parents=True, exist_ok=True)

HIP_LABELS = {"L": 17, "R": 53}


def zscore(x):
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(x)

    if good.sum() < 5:
        return y

    sd = np.nanstd(x[good])
    if not np.isfinite(sd) or sd == 0:
        return y

    y[good] = (x[good] - np.nanmean(x[good])) / sd
    return y


def residualize(y, nuisance):
    y = np.asarray(y, float)
    nuisance = np.asarray(nuisance, float)

    n = min(len(y), len(nuisance))
    y = y[:n]
    nuisance = nuisance[:n]

    good = np.isfinite(y) & np.isfinite(nuisance)
    out = np.full(n, np.nan, dtype=float)

    if good.sum() < 10:
        return out

    X = np.column_stack([
        np.ones(good.sum()),
        nuisance[good],
    ])

    beta, _, _, _ = np.linalg.lstsq(X, y[good], rcond=None)
    out[good] = y[good] - X @ beta

    return out


def corr(x, y):
    n = min(len(x), len(y))
    x = zscore(x[:n])
    y = zscore(y[:n])

    good = np.isfinite(x) & np.isfinite(y)

    if good.sum() < 10:
        return np.nan, good.sum()

    r, _ = stats.pearsonr(x[good], y[good])
    return r, good.sum()


def fisher_z(r):
    if not np.isfinite(r) or abs(r) >= 1:
        return np.nan
    return np.arctanh(r)


def find_boundary_file(subject, run):
    r = int(str(run).replace("run-", ""))

    candidates = [
        BOUNDARY_DIR / f"{subject}_run-{r:02d}.npy",
        BOUNDARY_DIR / f"{subject}_run-{r}.npy",
    ]

    for f in candidates:
        if f.exists():
            return f

    hits = list(BOUNDARY_DIR.glob(f"*{subject}*run-{r:02d}*.npy"))
    hits += list(BOUNDARY_DIR.glob(f"*{subject}*run-{r}*.npy"))

    if hits:
        return sorted(hits)[0]

    return None


def window_mask(centers, n, radius):
    mask = np.zeros(n, dtype=bool)

    for c in centers:
        lo = max(0, c - radius)
        hi = min(n, c + radius + 1)
        mask[lo:hi] = True

    return mask


def sample_random_centers(valid_centers, n_centers, rng):
    if len(valid_centers) < n_centers:
        return None
    return rng.choice(valid_centers, size=n_centers, replace=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=5, help="TR radius around boundary")
    ap.add_argument("--n-random", type=int, default=100, help="random control samples per run")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(IN)
    df = df[df["n_trs"] >= 100].copy()

    global_cache = {}
    hc_cache = {}

    rows = []

    for _, row in df.iterrows():
        subject = row["subject"]
        run = f"{int(row['run']):02d}"
        hemi = row["hipp_hemi"]
        vta_roi = row["vta_roi"]

        boundary_file = find_boundary_file(subject, run)

        if boundary_file is None:
            print(f"[skip] missing boundary: {subject} run-{run}")
            continue

        bold_file = row["bold_file"]
        aseg_file = row["aseg_file"]

        if bold_file not in global_cache:
            bold = nib.load(bold_file).get_fdata()
            global_cache[bold_file] = np.nanmean(
                bold.reshape(-1, bold.shape[-1]),
                axis=0,
            )

        key = (bold_file, aseg_file, hemi)

        if key not in hc_cache:
            bold = nib.load(bold_file).get_fdata()
            aseg = nib.load(aseg_file).get_fdata()
            mask = aseg == HIP_LABELS[hemi]
            hc_cache[key] = np.nanmean(bold[mask, :], axis=0)

        vta = np.load(row["vta_file"]).squeeze().astype(float)
        hc = hc_cache[key]
        global_ts = global_cache[bold_file]
        boundary = np.load(boundary_file).squeeze().astype(float)

        n = min(len(vta), len(hc), len(global_ts), len(boundary))

        vta = vta[:n]
        hc = hc[:n]
        global_ts = global_ts[:n]
        boundary = boundary[:n]

        vta_resid = residualize(vta, global_ts)
        hc_resid = residualize(hc, global_ts)

        centers = np.where(boundary > 0.5)[0]
        centers = centers[(centers >= args.radius) & (centers < n - args.radius)]

        if len(centers) < 3:
            print(f"[skip] too few boundaries: {subject} run-{run}")
            continue

        bmask = window_mask(centers, n, args.radius)

        r_boundary, n_boundary_samples = corr(vta_resid[bmask], hc_resid[bmask])
        z_boundary = fisher_z(r_boundary)

        # Random windows: avoid actual boundary centers.
        all_valid = np.arange(args.radius, n - args.radius)
        non_boundary_centers = np.setdiff1d(all_valid, centers)

        random_z = []

        for _ in range(args.n_random):
            sampled = sample_random_centers(
                non_boundary_centers,
                len(centers),
                rng,
            )

            if sampled is None:
                continue

            rmask = window_mask(sampled, n, args.radius)
            r_rand, _ = corr(vta_resid[rmask], hc_resid[rmask])
            z_rand = fisher_z(r_rand)

            if np.isfinite(z_rand):
                random_z.append(z_rand)

        random_z = np.asarray(random_z, float)

        if len(random_z) == 0:
            continue

        z_random_mean = np.nanmean(random_z)
        z_random_sd = np.nanstd(random_z)

        rows.append({
            "subject": subject,
            "run": run,
            "vta_roi": vta_roi,
            "hipp_hemi": hemi,
            "radius_tr": args.radius,
            "n_boundaries": len(centers),
            "n_boundary_samples": n_boundary_samples,
            "r_boundary": r_boundary,
            "z_boundary": z_boundary,
            "z_random_mean": z_random_mean,
            "r_random_mean_approx": np.tanh(z_random_mean),
            "z_random_sd": z_random_sd,
            "delta_z_boundary_minus_random": z_boundary - z_random_mean,
            "delta_r_approx": np.tanh(z_boundary) - np.tanh(z_random_mean),
            "boundary_file": str(boundary_file),
            "vta_file": row["vta_file"],
            "bold_file": bold_file,
            "aseg_file": aseg_file,
        })

        print(
            f"✅ {subject} run-{run} {vta_roi} HC-{hemi}: "
            f"boundary r={r_boundary:.3f}, "
            f"random r≈{np.tanh(z_random_mean):.3f}, "
            f"delta_z={z_boundary - z_random_mean:.3f}"
        )

    out = pd.DataFrame(rows)

    by_run_csv = OUT / "boundary_modulated_vta_hipp_coupling_by_run.csv"
    out.to_csv(by_run_csv, index=False)

    subj = (
        out.groupby(["subject", "vta_roi", "hipp_hemi"], as_index=False)
        .agg(
            delta_z_subject=("delta_z_boundary_minus_random", "mean"),
            z_boundary_subject=("z_boundary", "mean"),
            z_random_subject=("z_random_mean", "mean"),
            n_runs=("run", "nunique"),
        )
    )

    subj_csv = OUT / "boundary_modulated_vta_hipp_coupling_subject.csv"
    subj.to_csv(subj_csv, index=False)

    group_rows = []

    for (vta_roi, hemi), g in subj.groupby(["vta_roi", "hipp_hemi"]):
        vals = g["delta_z_subject"].dropna().values
        tval, pval = stats.ttest_1samp(vals, 0.0)

        group_rows.append({
            "vta_roi": vta_roi,
            "hipp_hemi": hemi,
            "n_subjects": len(vals),
            "mean_delta_z": vals.mean(),
            "sem_delta_z": stats.sem(vals),
            "mean_delta_r_approx": np.tanh(vals.mean()),
            "t": tval,
            "p": pval,
            "mean_runs": g["n_runs"].mean(),
        })

    group = pd.DataFrame(group_rows)

    group_csv = OUT / "boundary_modulated_vta_hipp_coupling_group.csv"
    group.to_csv(group_csv, index=False)

    print(f"\nWrote {by_run_csv}")
    print(f"Wrote {subj_csv}")
    print(f"Wrote {group_csv}")
    print(group)


if __name__ == "__main__":
    main()