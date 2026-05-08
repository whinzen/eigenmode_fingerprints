python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = Path.home() / "eigenmode_fingerprints" / "pang_out"

# Try both possible group folders
for cand in ["group_summary", "group"]:
    group_dir = base / cand
    if (group_dir / "group_onset_hemi-L.csv").exists():
        break
else:
    raise FileNotFoundError("Could not find group CSVs in pang_out/group_summary or pang_out/group")

# Optional energy for normalization (if present)
Emean_path = None
for cand in ["group_summary", "group"]:
    p = base / cand / "Emean_group.npy"
    if p.exists():
        Emean_path = p
        break
Emean_group = np.load(Emean_path) if Emean_path else None

def make_plot(kind="onset", hemi="L"):
    csv_path = group_dir / f"group_{kind}_hemi-{hemi}.csv"
    if not csv_path.exists():
        print(f"[skip] {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Expecting columns: mode_k, lam, beta_mean, beta_sem, sig_q05 (and maybe p_mean)
    # Drop mode 0
    df = df[df["mode_k"] > 0].copy()
    df.sort_values("mode_k", inplace=True)

    # ===== Plot 1: β vs mode index (k) with SEM, mark sig =====
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(df["mode_k"], df["beta_mean"], lw=1.8, label="β (mean)")
    ax.fill_between(df["mode_k"],
                    df["beta_mean"] - df["beta_sem"],
                    df["beta_mean"] + df["beta_sem"],
                    alpha=0.25, label="± SEM")
    sig = df["sig_q05"] == 1
    ax.scatter(df.loc[sig, "mode_k"], df.loc[sig, "beta_mean"], s=20, color="red", label="FDR q<0.05")
    ax.set_yscale("symlog")  # handles +/- and compresses dynamic range
    ax.set_xlabel("Eigenmode index (k)")
    ax.set_ylabel("Boundary effect β")
    ax.set_title(f"Sentence {kind} — hemi {hemi} (β vs k)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out1 = group_dir / f"group_{kind}_hemi-{hemi}_beta_vs_index_symlog.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # ===== Plot 2: β vs eigenvalue λ (log10 λ on x) =====
    lam = df["lam"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(np.log10(lam), df["beta_mean"], lw=1.8, label="β (mean)")
    ax.fill_between(np.log10(lam),
                    df["beta_mean"] - df["beta_sem"],
                    df["beta_mean"] + df["beta_sem"],
                    alpha=0.25, label="± SEM")
    ax.scatter(np.log10(lam[sig.values]), df.loc[sig, "beta_mean"], s=20, color="red", label="FDR q<0.05")
    ax.set_yscale("symlog")
    ax.set_xlabel("log10 eigenvalue λ  (higher → shorter wavelength)")
    ax.set_ylabel("Boundary effect β")
    ax.set_title(f"Sentence {kind} — hemi {hemi} (β vs log10 λ)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out2 = group_dir / f"group_{kind}_hemi-{hemi}_beta_vs_loglam_symlog.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)

    # ===== Optional: β normalized by mean energy (if available) =====
    if Emean_group is not None:
        # Align Emean_group by mode index; drop k=0 and clip to df rows
        # Assume Emean_group is a 1D array indexed by k (length >= max k+1)
        k = df["mode_k"].astype(int).to_numpy()
        if Emean_group.shape[0] > k.max():
            Enorm = Emean_group[k]  # k>0
            Enorm[Enorm == 0] = np.nan
            beta_norm = df["beta_mean"].to_numpy() / Enorm
            beta_norm_sem = df["beta_sem"].to_numpy() / Enorm

            # vs k
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            ax.plot(df["mode_k"], beta_norm, lw=1.8, label="β / Ē")
            ax.fill_between(df["mode_k"],
                            beta_norm - beta_norm_sem,
                            beta_norm + beta_norm_sem,
                            alpha=0.25, label="± SEM")
            ax.scatter(df.loc[sig, "mode_k"], beta_norm[sig.values], s=20, color="red", label="FDR q<0.05")
            ax.set_yscale("symlog")
            ax.set_xlabel("Eigenmode index (k)")
            ax.set_ylabel("Boundary effect normalized by mean energy (β / Ē)")
            ax.set_title(f"Sentence {kind} — hemi {hemi} (β/Ē vs k)")
            ax.legend(frameon=False)
            fig.tight_layout()
            out3 = group_dir / f"group_{kind}_hemi-{hemi}_betaNorm_vs_index_symlog.png"
            fig.savefig(out3, dpi=150)
            plt.close(fig)

            # vs log10 λ
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            ax.plot(np.log10(lam), beta_norm, lw=1.8, label="β / Ē")
            ax.fill_between(np.log10(lam),
                            beta_norm - beta_norm_sem,
                            beta_norm + beta_norm_sem,
                            alpha=0.25, label="± SEM")
            ax.scatter(np.log10(lam[sig.values]), beta_norm[sig.values], s=20, color="red", label="FDR q<0.05")
            ax.set_yscale("symlog")
            ax.set_xlabel("log10 eigenvalue λ  (higher → shorter wavelength)")
            ax.set_ylabel("Boundary effect normalized by mean energy (β / Ē)")
            ax.set_title(f"Sentence {kind} — hemi {hemi} (β/Ē vs log10 λ)")
            ax.legend(frameon=False)
            fig.tight_layout()
            out4 = group_dir / f"group_{kind}_hemi-{hemi}_betaNorm_vs_loglam_symlog.png"
            fig.savefig(out4, dpi=150)
            plt.close(fig)
        else:
            print(f"[warn] Emean_group length {Emean_group.shape[0]} < max k {k.max()} — skipping normalized plots.")

    print(f"[ok] Wrote:\n  {out1}\n  {out2}" + ("" if Emean_group is None else f"\n  (and normalized variants if Emean_group was found)"))

for kind in ["onset", "offset"]:
    for hemi in ["L", "R"]:
        make_plot(kind, hemi)
PY