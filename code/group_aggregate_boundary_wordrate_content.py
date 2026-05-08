#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"

OUT_DIR = BASE / "group_boundary_wordrate_content_glm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WAVELENGTH_TABLE = BASE / "group" / "wavelength_table.csv"

MODEL_NAME = "boundary_plus_wordrate_content"


def bh_fdr(p, alpha=0.05):
    p = np.asarray(p, float)
    sig = np.zeros(len(p), dtype=bool)

    mask = np.isfinite(p)
    if mask.sum() == 0:
        return sig

    idx = np.where(mask)[0]
    ps = p[mask]
    order = np.argsort(ps)

    thr = alpha * (np.arange(1, len(ps) + 1) / len(ps))
    below = ps[order] <= thr

    if below.any():
        kmax = np.where(below)[0].max()
        sig[idx[order[:kmax + 1]]] = True

    return sig


def load_lambda_table():
    t = pd.read_csv(WAVELENGTH_TABLE)

    if "mode_k" not in t.columns and "k" in t.columns:
        t = t.rename(columns={"k": "mode_k"})

    if "lam" not in t.columns and "lambda" in t.columns:
        t = t.rename(columns={"lambda": "lam"})

    if "mode_k" not in t.columns or "lam" not in t.columns:
        raise RuntimeError(
            f"wavelength table missing required columns: {list(t.columns)}"
        )

    return t[["mode_k", "lam"]].drop_duplicates("mode_k")


def collect_subject_wide():
    rows = []
    skipped_missing = 0
    skipped_empty = 0

    for sub in sorted(BASE.glob("sub-*")):
        if not sub.is_dir():
            continue

        f = sub / "glm_boundary_wordrate_content" / "boundary_wordrate_content_wide.csv"

        if not f.exists():
            skipped_missing += 1
            continue

        try:
            if f.stat().st_size == 0:
                print(f"[skip empty] {f}")
                skipped_empty += 1
                continue

            df = pd.read_csv(f)

            if df.empty or len(df.columns) == 0:
                print(f"[skip empty] {f}")
                skipped_empty += 1
                continue

        except pd.errors.EmptyDataError:
            print(f"[skip empty] {f}")
            skipped_empty += 1
            continue

        df["subject"] = sub.name
        df["metric"] = "boundary"
        df["control_model"] = MODEL_NAME
        rows.append(df)

    print(
        f"[controlled boundary] loaded subjects: {len(rows)} | "
        f"missing: {skipped_missing} | empty: {skipped_empty}"
    )

    if not rows:
        raise ValueError("No non-empty controlled boundary files found")

    return pd.concat(rows, ignore_index=True)


def aggregate_controlled_boundary():
    lam_table = load_lambda_table()
    big = collect_subject_wide()

    out_big = OUT_DIR / "group_boundary_wordrate_content_allruns.csv"
    big.to_csv(out_big, index=False)

    results = []

    for hemi in ["L", "R"]:
        d_hemi = big[big["hemi"] == hemi].copy()

        # Average within subject across runs first.
        # The compatibility column "beta" is beta_boundary.
        subj_mean = (
            d_hemi.groupby(["subject", "mode_k"], as_index=False)
            .agg(beta_mean_subj=("beta", "mean"))
        )

        for mode_k, g in subj_mean.groupby("mode_k"):
            vals = g["beta_mean_subj"].dropna().values

            if len(vals) < 2:
                results.append({
                    "metric": "boundary",
                    "control_model": MODEL_NAME,
                    "hemi": hemi,
                    "mode_k": int(mode_k),
                    "beta_mean": np.nan,
                    "beta_sem": np.nan,
                    "N_subj": len(vals),
                    "t": np.nan,
                    "p": np.nan,
                })
                continue

            t, p = ttest_1samp(vals, 0.0, nan_policy="omit")

            results.append({
                "metric": "boundary",
                "control_model": MODEL_NAME,
                "hemi": hemi,
                "mode_k": int(mode_k),
                "beta_mean": vals.mean(),
                "beta_sem": vals.std(ddof=1) / np.sqrt(len(vals)),
                "N_subj": len(vals),
                "t": t,
                "p": p,
            })

    out = pd.DataFrame(results)
    out = out.merge(lam_table, on="mode_k", how="left")

    for hemi in ["L", "R"]:
        mask = out["hemi"] == hemi
        out.loc[mask, "sig_q05"] = bh_fdr(
            out.loc[mask, "p"].values,
            alpha=0.05,
        ).astype(int)

    out["sig_q05"] = out["sig_q05"].astype("Int64")

    out_csv = OUT_DIR / "group_boundary_wordrate_content_by_mode_subject_level.csv"
    out.to_csv(out_csv, index=False)

    print(f"✅ wrote {out_big}")
    print(f"✅ wrote {out_csv}")

    return out


def main():
    aggregate_controlled_boundary()


if __name__ == "__main__":
    main()