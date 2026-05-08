# ~/eigenmode_fingerprints/code/subject_level_aggregate.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from settings import PANG_OUT, HEMIS, WAVELENGTH_TABLE

BASE = PANG_OUT / "group_boundary_glm"
BASE.mkdir(parents=True, exist_ok=True)

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
    mode_col = "mode_k" if "mode_k" in t.columns else ("mode" if "mode" in t.columns else None)
    lam_col  = "lambda" if "lambda" in t.columns else ("lam" if "lam" in t.columns else None)
    if not mode_col or not lam_col:
        raise RuntimeError("wavelength_table.csv missing 'mode/mode_k' or 'lambda/lam'")
    t = t.rename(columns={mode_col:"mode_k", lam_col:"lam"}).drop_duplicates("mode_k")
    return t[["mode_k","lam"]]

def load_perrun(hemi):
    df = pd.read_csv(BASE/f"group_onset_perrun_hemi-{hemi}.csv")
    df = df[np.isfinite(df["beta"]) & np.isfinite(df["mode_k"])]
    return df

def subject_level(df, lam_table, exclude_k0):
    df = df.merge(lam_table, on="mode_k", how="left")
    if exclude_k0:
        df = df[df["mode_k"] > 0]
    df = df[np.isfinite(df["lam"])]

    # within-subject average across runs
    sub_avg = (df.groupby(["subject","mode_k"], as_index=False)
                 .agg(beta=("beta","mean")))
    sub_avg = sub_avg.merge(lam_table, on="mode_k", how="left")

    # across-subject summary
    g = (sub_avg.groupby("mode_k", as_index=False)
         .agg(lam=("lam","median"),
              beta_mean=("beta","mean"),
              beta_sem=("beta", lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan),
              N=("beta","size")))

    # simple two-sided t-test against 0 using summary stats
    # t = mean / (sem) ; df = N-1 ; approximate p via normal
    # (for robust reproducibility without SciPy)
    z = g["beta_mean"] / g["beta_sem"]
    # approximate two-tailed p via normal
    from math import erf, sqrt
    def p_norm2(z):
        if not np.isfinite(z): return np.nan
        return 2.0*(1.0 - 0.5*(1.0+erf(abs(z)/sqrt(2.0))))
    g["p"] = [p_norm2(float(zz)) for zz in z]
    g["sig_q05"] = bh_fdr(g["p"].values, alpha=0.05).astype(int)

    return g.sort_values("mode_k"), sub_avg

def plot(g, hemi, out_png, title):
    x = g["mode_k"].to_numpy()
    y = g["beta_mean"].to_numpy()
    e = g["beta_sem"].to_numpy()
    sig = g["sig_q05"].to_numpy(dtype=bool)

    plt.figure(figsize=(7,4))
    plt.plot(x, y, lw=2)
    if np.isfinite(e).any():
        plt.fill_between(x, y-e, y+e, alpha=0.2)
    if sig.any():
        plt.scatter(x[sig], y[sig], s=18)
    plt.axhline(0, ls="--", lw=1)
    plt.xlabel("Eigenmode index k")
    plt.ylabel("β (sentence boundary)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"✅ saved {out_png}")

def main():
    lam_table = load_lambda_table()
    for hemi in HEMIS:
        df = load_perrun(hemi)
        g0, _ = subject_level(df, lam_table, exclude_k0=False)
        g1, _ = subject_level(df, lam_table, exclude_k0=True)

        f0 = BASE/f"group_onset_hemi-{hemi}_subjectlevel_by_mode.csv"
        f1 = BASE/f"group_onset_hemi-{hemi}_subjectlevel_by_mode_excl_k0.csv"
        g0.to_csv(f0, index=False); g1.to_csv(f1, index=False)

        plot(g0, hemi, BASE/f"onset_subjectmean_{hemi}_byindex_incl_k0.png",
             f"Subject mean β (incl k=0) – hemi {hemi}")
        plot(g1, hemi, BASE/f"onset_subjectmean_{hemi}_byindex_excl_k0.png",
             f"Subject mean β (excl k=0) – hemi {hemi}")

if __name__ == "__main__":
    main()