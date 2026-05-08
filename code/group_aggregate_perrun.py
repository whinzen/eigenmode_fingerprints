# ~/eigenmode_fingerprints/code/group_aggregate_perrun.py
from pathlib import Path
import numpy as np
import pandas as pd
from settings import PANG_OUT, HEMIS, WAVELENGTH_TABLE, GLM_PER_RUN_DIR

OUT_DIR = PANG_OUT / "group_boundary_glm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def bh_fdr(p, alpha=0.05):
    p = np.asarray(p, float)
    mask = np.isfinite(p)
    sig = np.zeros(len(p), bool)
    if mask.sum() == 0: return sig
    idx = np.where(mask)[0]
    ps = p[mask]
    order = np.argsort(ps)
    thr = alpha * (np.arange(1, len(ps)+1) / len(ps))
    below = ps[order] <= thr
    if below.any():
        kmax = np.where(below)[0].max()
        sig[idx[order[:kmax+1]]] = True
    return sig

def load_lambda_table():
    t = pd.read_csv(WAVELENGTH_TABLE)
    # tolerate variant column names
    mode_col = "mode_k" if "mode_k" in t.columns else ("mode" if "mode" in t.columns else None)
    lam_col  = "lambda" if "lambda" in t.columns else ("lam" if "lam" in t.columns else None)
    if not mode_col or not lam_col:
        raise RuntimeError("wavelength_table.csv missing 'mode'/'mode_k' or 'lambda/lam' columns.")
    t = t.rename(columns={mode_col: "mode_k", lam_col: "lam"}).drop_duplicates("mode_k")
    return t[["mode_k","lam"]]

def collect_perrun(hemi):
    rows = []
    for sdir in sorted((PANG_OUT).glob("sub-*")):
        per = sdir / GLM_PER_RUN_DIR
        if not per.is_dir(): continue
        files = sorted(per.glob(f"onset_{hemi}_run-*.csv"))
        for f in files:
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            rows.append(df)
    if not rows:
        raise SystemExit(f"❌ no per-run files for hemi={hemi}")
    big = pd.concat(rows, ignore_index=True)
    return big

def by_mode(big, lam_table, exclude_k0=True):
    # merge λ
    big = big.merge(lam_table, on="mode_k", how="left")
    if exclude_k0:
        big = big[big["mode_k"] > 0]
    big = big[np.isfinite(big["beta"]) & np.isfinite(big["lam"])]

    g = big.groupby("mode_k", as_index=False).agg(
        lam=("lam","median"),
        beta_mean=("beta","mean"),
        beta_sem=("beta", lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan),
        N=("beta","size"),
        p_median=("p","median")
    )
    g["sig_q05"] = bh_fdr(g["p_median"].values, alpha=0.05).astype(int)
    return g.sort_values("mode_k")

def main():
    lam_table = load_lambda_table()

    for hemi in HEMIS:
        big = collect_perrun(hemi)
        out_big = OUT_DIR / f"group_onset_perrun_hemi-{hemi}.csv"
        big.to_csv(out_big, index=False)
        print(f"✅ wrote {out_big}  rows={len(big):,}")

        g = by_mode(big, lam_table, exclude_k0=True)
        out_sm = OUT_DIR / f"group_onset_hemi-{hemi}_by_mode_excl_k0.csv"
        g.to_csv(out_sm, index=False)
        print(f"✅ wrote {out_sm}  modes={len(g)}")

if __name__ == "__main__":
    main()