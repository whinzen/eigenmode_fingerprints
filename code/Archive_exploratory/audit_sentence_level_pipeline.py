from pathlib import Path
import pandas as pd
import numpy as np
import re

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"
CODE = BASE / "code"

FILES = {
    "boundary_group_L_old": PANG / "group_boundary_glm" / "group_onset_hemi-L_subjectlevel_by_mode_excl_k0.csv",
    "boundary_group_R_old": PANG / "group_boundary_glm" / "group_onset_hemi-R_subjectlevel_by_mode_excl_k0.csv",
    "shift_group_L": PANG / "group_sentence_shift_glm" / "group_sentence_shift_hemi-L_by_mode_subject_level.csv",
    "shift_group_R": PANG / "group_sentence_shift_glm" / "group_sentence_shift_hemi-R_by_mode_subject_level.csv",
}

SCRIPT_CANDIDATES = [
    CODE / "rebuild_glm_per_run.py",
    CODE / "rebuild_glm_sentence_shift_per_run.py",
    CODE / "group_aggregate_perrun.py",
    CODE / "group_aggregate_sentence_shift.py",
    CODE / "settings.py",
]

def inspect_csv(path: Path):
    if not path.exists():
        return {"exists": False}
    df = pd.read_csv(path)
    out = {
        "exists": True,
        "rows": len(df),
        "cols": list(df.columns),
    }
    if "mode_k" in df.columns:
        out["n_unique_modes"] = df["mode_k"].nunique()
        out["mode_min"] = int(df["mode_k"].min())
        out["mode_max"] = int(df["mode_k"].max())
    elif "k" in df.columns:
        out["n_unique_modes"] = df["k"].nunique()
        out["mode_min"] = int(df["k"].min())
        out["mode_max"] = int(df["k"].max())
    if "sig_q05" in df.columns:
        out["n_sig"] = int((df["sig_q05"] == 1).sum())
    return out

def grep_text(path: Path, patterns):
    if not path.exists():
        return {"exists": False}
    txt = path.read_text(errors="ignore")
    hits = {}
    for pat in patterns:
        m = re.findall(pat, txt, flags=re.MULTILINE)
        hits[pat] = m[:10]
    return {"exists": True, "hits": hits}

def main():
    print("=== CSV AUDIT ===")
    for name, path in FILES.items():
        print(f"\n[{name}] {path}")
        info = inspect_csv(path)
        for k, v in info.items():
            print(f"  {k}: {v}")

    print("\n=== SCRIPT AUDIT ===")
    pats = [
        r"TR_SEC\s*=\s*.*",
        r"MIN_TRS\s*=\s*.*",
        r"K\s*=\s*\d+",
        r"kmodes\s*=\s*\d+",
        r"A_[LR]\.npy",
        r"E_k_t",
        r"sentence_shift",
        r"boundary",
        r"p_unc",
        r"sig_q05",
    ]
    for path in SCRIPT_CANDIDATES:
        print(f"\n[{path.name}]")
        info = grep_text(path, pats)
        if not info["exists"]:
            print("  missing")
            continue
        for pat, vals in info["hits"].items():
            if vals:
                print(f"  {pat}: {vals}")

if __name__ == "__main__":
    main()