#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN = PANG / "subcortex" / "vta_hipp_coupling" / "vta_hipp_timeseries_coupling_by_run.csv"
OUT = PANG / "subcortex" / "vta_hipp_coupling_global_residualized"
OUT.mkdir(parents=True, exist_ok=True)

HIP_LABELS = {"L": 17, "R": 53}


def zscore(x):
    x = np.asarray(x, float)
    good = np.isfinite(x)
    y = np.full_like(x, np.nan, dtype=float)

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
    resid = y[good] - X @ beta

    out[good] = resid
    return out


def corr(x, y):
    n = min(len(x), len(y))
    x = zscore(x[:n])
    y = zscore(y[:n])

    good = np.isfinite(x) & np.isfinite(y)

    if good.sum() < 10:
        return np.nan, np.nan, good.sum()

    r, p = stats.pearsonr(x[good], y[good])
    return r, p, good.sum()


def fisher_z(r):
    if not np.isfinite(r) or abs(r) >= 1:
        return np.nan
    return np.arctanh(r)


def main():
    df = pd.read_csv(IN)
    df = df[df["n_trs"] >= 100].copy()

    global_cache = {}
    hc_cache = {}

    rows = []

    for _, row in df.iterrows():
        bold_file = row["bold_file"]
        aseg_file = row["aseg_file"]
        hemi = row["hipp_hemi"]

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

        global_ts = global_cache[bold_file]
        hc = hc_cache[key]
        vta = np.load(row["vta_file"]).squeeze().astype(float)

        n = min(len(vta), len(hc), len(global_ts))

        vta = vta[:n]
        hc = hc[:n]
        global_ts = global_ts[:n]

        r_raw, p_raw, n_used = corr(vta, hc)

        vta_resid = residualize(vta, global_ts)
        hc_resid = residualize(hc, global_ts)

        r_resid, p_resid, n_resid = corr(vta_resid, hc_resid)

        dvta = np.diff(vta_resid)
        dhc = np.diff(hc_resid)
        r_diff_resid, p_diff_resid, n_diff = corr(dvta, dhc)

        rows.append({
            "subject": row["subject"],
            "run": f"{int(row['run']):02d}",
            "vta_roi": row["vta_roi"],
            "hipp_hemi": hemi,
            "r_raw": r_raw,
            "z_raw": fisher_z(r_raw),
            "p_raw": p_raw,
            "r_global_resid": r_resid,
            "z_global_resid": fisher_z(r_resid),
            "p_global_resid": p_resid,
            "r_diff_global_resid": r_diff_resid,
            "z_diff_global_resid": fisher_z(r_diff_resid),
            "p_diff_global_resid": p_diff_resid,
            "n_used": n_used,
            "n_resid": n_resid,
            "n_diff": n_diff,
            "vta_file": row["vta_file"],
            "bold_file": bold_file,
            "aseg_file": aseg_file,
        })

        print(
            f"✅ {row['subject']} run-{int(row['run']):02d} "
            f"{row['vta_roi']} HC-{hemi}: "
            f"raw={r_raw:.3f}, resid={r_resid:.3f}, diff_resid={r_diff_resid:.3f}"
        )

    out = pd.DataFrame(rows)

    by_run_csv = OUT / "vta_hipp_coupling_global_residualized_by_run.csv"
    out.to_csv(by_run_csv, index=False)

    subj = (
        out.groupby(["subject", "vta_roi", "hipp_hemi"], as_index=False)
        .agg(
            z_raw_subject=("z_raw", "mean"),
            z_global_resid_subject=("z_global_resid", "mean"),
            z_diff_global_resid_subject=("z_diff_global_resid", "mean"),
            n_runs=("run", "nunique"),
        )
    )

    subj_csv = OUT / "vta_hipp_coupling_global_residualized_subject.csv"
    subj.to_csv(subj_csv, index=False)

    group_rows = []

    for (vta_roi, hemi), g in subj.groupby(["vta_roi", "hipp_hemi"]):
        for col, label in [
            ("z_raw_subject", "raw"),
            ("z_global_resid_subject", "global_residualized"),
            ("z_diff_global_resid_subject", "diff_global_residualized"),
        ]:
            vals = g[col].dropna().values
            tval, pval = stats.ttest_1samp(vals, 0.0)

            group_rows.append({
                "vta_roi": vta_roi,
                "hipp_hemi": hemi,
                "coupling_type": label,
                "n_subjects": len(vals),
                "mean_z": vals.mean(),
                "sem_z": stats.sem(vals),
                "mean_r_approx": np.tanh(vals.mean()),
                "t": tval,
                "p": pval,
                "mean_runs": g["n_runs"].mean(),
            })

    group = pd.DataFrame(group_rows)

    group_csv = OUT / "vta_hipp_coupling_global_residualized_group.csv"
    group.to_csv(group_csv, index=False)

    print(f"\nWrote {by_run_csv}")
    print(f"Wrote {subj_csv}")
    print(f"Wrote {group_csv}")
    print(group)


if __name__ == "__main__":
    main()