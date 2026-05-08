import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gamma

BASE = Path.home() / "eigenmode_fingerprints"
CSV_PATH = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"
PANG_OUT = BASE / "pang_out"

REG_DIR = PANG_OUT / "regressors" / "content_density_hrf_per_subject"
REG_DIR.mkdir(parents=True, exist_ok=True)

TR = 2.0

FUNCTION_WORDS = {
    "a", "an", "the",
    "and", "or", "but", "if", "because", "as", "while", "although",
    "of", "in", "on", "at", "to", "from", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very",
    "can", "will", "just", "should", "now",
    "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom",
    "this", "that", "these", "those",
    "am", "is", "are", "was", "were",
    "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "would", "could", "should", "may", "might", "must", "shall",
}


def canonical_hrf(tr, length=32.0):
    t = np.arange(0, length, tr)
    peak = gamma.pdf(t, 6)
    undershoot = gamma.pdf(t, 16)
    hrf = peak - 0.5 * undershoot
    if np.max(np.abs(hrf)) > 0:
        hrf = hrf / np.max(np.abs(hrf))
    return hrf


def zscore_safe(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    s = x.std()
    if s == 0:
        return x - x.mean()
    return (x - x.mean()) / s


def get_subject_runs(sub_dir: Path):
    runs = []
    for rdir in sorted(sub_dir.glob("run-*")):
        if not rdir.is_dir():
            continue
        try:
            run = int(rdir.name.split("-")[1])
            runs.append(run)
        except Exception:
            continue
    return sorted(runs)


def is_content_word(word):
    w = str(word).strip().lower()

    if not w:
        return 0

    # Ignore pure punctuation / non-lexical tokens
    if not re.search(r"[a-zA-Z]", w):
        return 0

    return int(w not in FUNCTION_WORDS)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing transcript CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = ["word", "run_id", "snt_id", "token_id", "onset", "offset"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {CSV_PATH}: {missing}")

    df = df.copy()
    df["word"] = df["word"].astype(str)
    df["run_id"] = df["run_id"].astype(int)
    df["snt_id"] = df["snt_id"].astype(int)
    df["token_id"] = df["token_id"].astype(int)
    df["onset"] = df["onset"].astype(float)
    df["offset"] = df["offset"].astype(float)

    df = df.sort_values(["run_id", "snt_id", "token_id"]).reset_index(drop=True)

    # Content-word indicator at each word onset.
    # This is not POS tagging proper; it is a conservative lexical-class proxy:
    # closed-class/function words = 0, content-like lexical items = 1.
    df["content_word"] = df["word"].map(is_content_word).astype(float)

    print("Content-word summary:")
    print(df["content_word"].value_counts(dropna=False).sort_index())

    hrf = canonical_hrf(TR)
    stim_regs = {}

    for run_id, g in df.groupby("run_id", sort=True):
        g = g.sort_values(["snt_id", "token_id"]).reset_index(drop=True)

        T = int(np.ceil(g["offset"].max() / TR)) + 1
        x = np.zeros(T, dtype=float)

        for onset, val in zip(g["onset"].values, g["content_word"].values):
            if not np.isfinite(onset) or not np.isfinite(val):
                continue
            idx = int(round(onset / TR))
            if 0 <= idx < T:
                x[idx] += float(val)

        x_hrf = np.convolve(x, hrf, mode="full")[:T]
        x_hrf = zscore_safe(x_hrf)

        stim_regs[int(run_id)] = x_hrf

    print(f"✅ Built stimulus-space content-density regressors for runs: {sorted(stim_regs.keys())}")

    subjects = sorted([p for p in PANG_OUT.glob("sub-*") if p.is_dir()])
    stim_run_ids = sorted(stim_regs.keys())

    for sub_dir in subjects:
        sub = sub_dir.name
        subj_runs = get_subject_runs(sub_dir)

        n = min(len(subj_runs), len(stim_run_ids))
        if n == 0:
            print(f"[skip] {sub}: no runs found")
            continue

        for i in range(n):
            subj_run = subj_runs[i]
            stim_run = stim_run_ids[i]

            out = REG_DIR / f"{sub}_run-{subj_run:02d}.npy"
            np.save(out, stim_regs[stim_run])

        print(f"✅ {sub}: wrote {n} subject-specific content-density regressors")

    print(f"✅ Done. Output directory: {REG_DIR}")


if __name__ == "__main__":
    main()