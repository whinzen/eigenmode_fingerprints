import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp

BASE = Path.home() / "eigenmode_fingerprints"
PANG_OUT = BASE / "pang_out"
WAVELENGTH_TABLE = PANG_OUT / "group" / "wavelength_table.csv"

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
    # tolerate variants
    if "mode_k" not in t.columns and "k" in t.columns:
        t = t.rename(columns={"k": "mode_k"})
    if "lam" not in t.columns and "lambda" in t.columns:
        t = t.rename(columns={"lambda": "lam"})
    if "mode_k" not in t.columns or "lam" not in t.columns:
        raise RuntimeError(f"wavelength table missing required columns. Found: {list(t.columns)}")
    return t[["mode_k", "lam"]].drop_duplicates("mode_k")

def collect_all_runs(metric, hemi):
    rows = []
    for sdir in sorted(PANG_OUT.glob("sub-*")):
        per = sdir / f"glm_{metric}" / "per_run"
        if not per.exists():
            continue
        for f in sorted(per.glob(f"{metric}_{hemi}_run-*.csv")):
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            rows.append(df)
    if not rows:
        raise SystemExit(f"❌ no per-run files found for metric={metric}, hemi={hemi}")
    return pd.concat(rows, ignore_index=True)

def aggregate_subject_level(big, lam_table, exclude_k0=True):
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
            "beta_mean": vals.mean(),
            "beta_sem": vals.std(ddof=1) / np.sqrt(len(vals)),
            "N_subj": len(vals),
            "t": t,
            "p": p,
        })

    out = pd.DataFrame(out)
    out = out.merge(lam_table, on="mode_k", how="left")
    out["sig_q05"] = bh_fdr(out["p"].values, alpha=0.05).astype(int)
    return out.sort_values("mode_k")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True)
    args = ap.parse_args()

    lam_table = load_lambda_table()
    out_dir = PANG_OUT / f"group_{args.metric}_glm"
    out_dir.mkdir(parents=True, exist_ok=True)

    for hemi in ["L", "R"]:
        big = collect_all_runs(args.metric, hemi)
        big_out = out_dir / f"group_{args.metric}_allruns_hemi-{hemi}.csv"
        big.to_csv(big_out, index=False)

        g = aggregate_subject_level(big, lam_table, exclude_k0=True)
        g_out = out_dir / f"group_{args.metric}_hemi-{hemi}_by_mode_subject_level.csv"
        g.to_csv(g_out, index=False)

        print(f"✅ wrote {big_out}")
        print(f"✅ wrote {g_out}")

if __name__ == "__main__":
    main()