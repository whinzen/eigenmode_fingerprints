#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"
DATA_ROOT = BASE / "data" / "empirical"

VTA_INDEX = PANG / "subcortex" / "brainstem_roi_timeseries" / "brainstem_roi_timeseries_index.csv"
OUT = PANG / "subcortex" / "vta_hipp_coupling"
OUT.mkdir(parents=True, exist_ok=True)

HIP_LABELS = {
    "L": 17,
    "R": 53,
}


def zscore(x):
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(x)

    if good.sum() < 3:
        return y

    sd = np.nanstd(x[good])
    if sd == 0 or not np.isfinite(sd):
        return y

    y[good] = (x[good] - np.nanmean(x[good])) / sd
    return y


def infer_run(path):
    for p in path.name.split("_"):
        if p.startswith("run-"):
            return p.replace("run-", "")
    raise ValueError(path)


def find_bold(subject, run):
    r = int(str(run).replace("run-", ""))
    hits = sorted(
        (DATA_ROOT / subject / "func").glob(
            f"*task-lppEN_run-{r:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
    )
    if hits:
        return hits[0]

    hits = sorted(
        (DATA_ROOT / subject / "func").glob(
            f"*task-lppEN_run-{r}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
    )
    if hits:
        return hits[0]

    return None


def hippocampal_mean_signal(subject, run, hemi):
    bold_file = find_bold(subject, run)

    if bold_file is None:
        return None, None, None

    aseg_file = Path(
        str(bold_file).replace(
            "_desc-preproc_bold.nii.gz",
            "_desc-aseg_dseg.nii.gz",
        )
    )

    if not aseg_file.exists():
        return None, bold_file, None

    bold = nib.load(str(bold_file)).get_fdata()
    aseg = nib.load(str(aseg_file)).get_fdata()

    label = HIP_LABELS[hemi]
    mask = aseg == label

    if mask.sum() < 20:
        return None, bold_file, aseg_file

    ts = np.nanmean(bold[mask, :], axis=0)

    return ts, bold_file, aseg_file


def corr_ts(x, y):
    n = min(len(x), len(y))
    x = zscore(x[:n])
    y = zscore(y[:n])

    good = np.isfinite(x) & np.isfinite(y)

    if good.sum() < 10:
        return np.nan, np.nan, good.sum()

    r, p = stats.pearsonr(x[good], y[good])

    return r, p, good.sum()


def main():
    idx = pd.read_csv(VTA_INDEX)

    # Focus on VTA first; LC is too small to treat as primary.
    idx = idx[idx["roi"].isin(["VTA_L", "VTA_R"])].copy()

    rows = []

    for _, row in idx.iterrows():
        subject = row["subject"]
        run = str(row["run"]).zfill(2)
        roi = row["roi"]

        vta_ts = np.load(row["timeseries_file"]).squeeze().astype(float)

        for hemi in ["L", "R"]:
            hc_ts, bold_file, aseg_file = hippocampal_mean_signal(subject, run, hemi)

            if hc_ts is None:
                print(f"[skip] missing HC: {subject} run-{run} hemi-{hemi}")
                continue

            r, p, n = corr_ts(vta_ts, hc_ts)

            z = np.arctanh(r) if np.isfinite(r) and abs(r) < 1 else np.nan

            rows.append({
                "subject": subject,
                "run": run,
                "vta_roi": roi,
                "hipp_hemi": hemi,
                "coupling_r": r,
                "coupling_z": z,
                "p": p,
                "n_trs": n,
                "vta_file": row["timeseries_file"],
                "bold_file": str(bold_file),
                "aseg_file": str(aseg_file),
            })

        print(f"✅ {subject} run-{run} {roi}")

    df = pd.DataFrame(rows)

    out_csv = OUT / "vta_hipp_timeseries_coupling_by_run.csv"
    df.to_csv(out_csv, index=False)

    # subject-level average Fisher z
    subj = (
        df.groupby(["subject", "vta_roi", "hipp_hemi"], as_index=False)
        .agg(
            coupling_z_subject=("coupling_z", "mean"),
            n_runs=("run", "nunique"),
        )
    )

    subj_csv = OUT / "vta_hipp_timeseries_coupling_subject.csv"
    subj.to_csv(subj_csv, index=False)

    group_rows = []

    for (vta_roi, hipp_hemi), g in subj.groupby(["vta_roi", "hipp_hemi"]):
        vals = g["coupling_z_subject"].dropna().values

        tval, pval = stats.ttest_1samp(vals, 0.0)

        group_rows.append({
            "vta_roi": vta_roi,
            "hipp_hemi": hipp_hemi,
            "n_subjects": len(vals),
            "mean_z": vals.mean(),
            "sem_z": stats.sem(vals),
            "mean_r_approx": np.tanh(vals.mean()),
            "t": tval,
            "p": pval,
            "mean_runs": g["n_runs"].mean(),
        })

    group = pd.DataFrame(group_rows)

    group_csv = OUT / "vta_hipp_timeseries_coupling_group.csv"
    group.to_csv(group_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(f"Wrote {subj_csv}")
    print(f"Wrote {group_csv}")
    print(group)


if __name__ == "__main__":
    main()