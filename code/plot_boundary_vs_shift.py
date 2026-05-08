from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from settings import PANG_OUT

BOUND_DIR = PANG_OUT / "group_boundary_glm"
SHIFT_DIR = PANG_OUT / "group_sentence_shift_glm"
OUT_DIR = PANG_OUT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_boundary(hemi):
    f = BOUND_DIR / f"group_onset_hemi-{hemi}_subjectlevel_by_mode_excl_k0.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing boundary file: {f}")
    df = pd.read_csv(f).copy()
    df["analysis"] = "boundary"
    return df

def load_shift(hemi):
    f = SHIFT_DIR / f"group_sentence_shift_hemi-{hemi}_by_mode_subject_level.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing sentence-shift file: {f}")
    df = pd.read_csv(f).copy()
    df["analysis"] = "sentence_shift"
    return df

def plot_vs_mode(hemi):
    b = load_boundary(hemi)
    s = load_shift(hemi)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(b["mode_k"], b["beta_mean"], linewidth=2, label="Sentence boundary")
    ax.fill_between(
        b["mode_k"],
        b["beta_mean"] - b["beta_sem"],
        b["beta_mean"] + b["beta_sem"],
        alpha=0.2
    )

    ax.plot(s["mode_k"], s["beta_mean"], linewidth=2, label="Sentence shift")
    ax.fill_between(
        s["mode_k"],
        s["beta_mean"] - s["beta_sem"],
        s["beta_mean"] + s["beta_sem"],
        alpha=0.2
    )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xlabel("Mode index k")
    ax.set_ylabel("β")
    ax.set_title(f"Boundary vs sentence shift ({hemi})")
    ax.legend(frameon=False)
    fig.tight_layout()

    out = OUT_DIR / f"boundary_vs_shift_hemi-{hemi}_by_mode.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✅ wrote {out}")

def plot_vs_lambda(hemi):
    b = load_boundary(hemi).sort_values("lam")
    s = load_shift(hemi).sort_values("lam")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(b["lam"], b["beta_mean"], linewidth=2, label="Sentence boundary")
    ax.fill_between(
        b["lam"],
        b["beta_mean"] - b["beta_sem"],
        b["beta_mean"] + b["beta_sem"],
        alpha=0.2
    )

    ax.plot(s["lam"], s["beta_mean"], linewidth=2, label="Sentence shift")
    ax.fill_between(
        s["lam"],
        s["beta_mean"] - s["beta_sem"],
        s["beta_mean"] + s["beta_sem"],
        alpha=0.2
    )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Eigenvalue λ")
    ax.set_ylabel("β")
    ax.set_title(f"Boundary vs sentence shift ({hemi})")
    ax.legend(frameon=False)
    fig.tight_layout()

    out = OUT_DIR / f"boundary_vs_shift_hemi-{hemi}_by_lambda.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✅ wrote {out}")

def main():
    for hemi in ["L", "R"]:
        plot_vs_mode(hemi)
        plot_vs_lambda(hemi)

if __name__ == "__main__":
    main()