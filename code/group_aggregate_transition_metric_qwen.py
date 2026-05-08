#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp


BASE = Path.home() / "eigenmode_fingerprints"
PANG_OUT = BASE / "pang_out"
WAVELENGTH_TABLE = PANG_OUT / "group" / "wavelength_table.csv"

QWEN_LABEL = "qwen3_0p6b"

METRICS = [
    "shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]


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
    if not WAVELENGTH_TABLE.exists():
        print(f"[warning] wavelength table not found: {WAVELENGTH_TABLE}")
        return pd.DataFrame({"mode_k": [], "lam": []})

    t = pd.read_csv(WAVELENGTH_TABLE)

    if "mode_k" not in t.columns and "k" in t.columns:
        t = t.rename(columns={"k": "mode_k"})

    if "lam" not in t.columns and "lambda" in t.columns:
        t = t.rename(columns={"lambda": "lam"})

    if "mode_k" not in t.columns or "lam" not in t.columns:
        raise RuntimeError(
            f"wavelength table missing required columns. Found: {list(t.columns)}"
        )

    t["mode_k"] = t["mode_k"].astype(int)
    return t[["mode_k", "lam"]].drop_duplicates("mode_k")


def collect_all_runs(metric, hemi):
    rows = []

    for sdir in sorted(PANG_OUT.glob("sub-*")):
        per = sdir / f"glm_{QWEN_LABEL}_{metric}" / "per_run"

        if not per.exists():
            continue

        pattern = f"{metric}_{hemi}_run-*.csv"

        for f in sorted(per.glob(pattern)):
            df = pd.read_csv(f)

            if df.empty:
                continue

            df["source_file"] = str(f)
            df["embedding_model"] = QWEN_LABEL
            df["metric"] = metric

            rows.append(df)

    if not rows:
        raise SystemExit(
            f"❌ no per-run files found for metric={metric}, hemi={hemi}. "
            f"Expected folders like pang_out/sub-*/glm_{QWEN_LABEL}_{metric}/per_run/"
        )

    return pd.concat(rows, ignore_index=True)


def aggregate_subject_level(big, lam_table, exclude_k0=True):
    big = big.copy()

    if exclude_k0:
        big = big[big["mode_k"] > 0].copy()

    subj_mean = (
        big.groupby(["subject", "mode_k"], as_index=False)
        .agg(beta=("beta", "mean"))
    )

    out = []

    for mode_k, g in subj_mean.groupby("mode_k"):
        vals = g["beta"].dropna().values

        if len(vals) < 2:
            out.append({
                "mode_k": int(mode_k),
                "beta_mean": np.nan,
                "beta_sem": np.nan,
                "N_subj": len(vals),
                "t": np.nan,
                "p": np.nan,
            })
            continue

        t, p = ttest_1samp(vals, 0.0, nan_policy="omit")

        out.append({
            "mode_k": int(mode_k),
            "beta_mean": float(vals.mean()),
            "beta_sem": float(vals.std(ddof=1) / np.sqrt(len(vals))),
            "N_subj": int(len(vals)),
            "t": float(t),
            "p": float(p),
        })

    out = pd.DataFrame(out)

    if not lam_table.empty:
        out = out.merge(lam_table, on="mode_k", how="left")
    else:
        out["lam"] = np.nan

    out["sig_q05"] = bh_fdr(out["p"].values, alpha=0.05).astype(int)
    out["embedding_model"] = QWEN_LABEL

    return out.sort_values("mode_k")


def run_one_metric(metric):
    lam_table = load_lambda_table()

    out_dir = PANG_OUT / f"group_{QWEN_LABEL}_{metric}_glm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Aggregating Qwen metric: {metric} ===")
    print(f"Output: {out_dir}")

    for hemi in ["L", "R"]:
        big = collect_all_runs(metric, hemi)

        big_out = out_dir / f"group_{QWEN_LABEL}_{metric}_allruns_hemi-{hemi}.csv"
        big.to_csv(big_out, index=False)

        g = aggregate_subject_level(big, lam_table, exclude_k0=True)

        g_out = (
            out_dir
            / f"group_{QWEN_LABEL}_{metric}_hemi-{hemi}_by_mode_subject_level.csv"
        )
        g.to_csv(g_out, index=False)

        print(f"✅ wrote {big_out} ({len(big)} rows)")
        print(f"✅ wrote {g_out} ({len(g)} modes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metric",
        default="all",
        choices=METRICS + ["all"],
        help="Metric to aggregate, or all.",
    )

    args = ap.parse_args()

    metrics = METRICS if args.metric == "all" else [args.metric]

    for metric in metrics:
        run_one_metric(metric)


if __name__ == "__main__":
    main()