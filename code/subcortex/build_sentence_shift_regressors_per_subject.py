#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.stats import gamma

BASE = Path.home() / "eigenmode_fingerprints"

CSV_PATH = (
    BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"
)

PANG_OUT = BASE / "pang_out"

PRE_DIR = PANG_OUT / "regressors" / "sentence_shift_per_subject"
HRF_DIR = PANG_OUT / "regressors" / "sentence_shift_hrf_per_subject"

PRE_DIR.mkdir(parents=True, exist_ok=True)
HRF_DIR.mkdir(parents=True, exist_ok=True)

TR = 2.0
MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def spm_hrf(tr, length=32.0):
    t = np.arange(0, length, tr)
    peak = gamma.pdf(t, 6)
    undershoot = gamma.pdf(t, 16)
    hrf = peak - 0.5 * undershoot

    if np.max(np.abs(hrf)) > 0:
        hrf = hrf / np.max(np.abs(hrf))

    return hrf


HRF = spm_hrf(TR)


def cosine_distance(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return 0.0

    return 1.0 - np.dot(a, b) / (na * nb)


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


print("Loading BERT model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

print("Loading transcript...")
df = pd.read_csv(CSV_PATH)

required = ["run_id", "snt_id", "token_id", "word", "onset", "offset"]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

df = df.sort_values(["run_id", "snt_id", "token_id"]).reset_index(drop=True)

# ------------------------------------------------------------------
# Build sentence embeddings
# ------------------------------------------------------------------

sentences = (
    df.groupby(["run_id", "snt_id"])["word"]
    .apply(list)
    .reset_index()
)

sentence_keys = list(zip(sentences["run_id"], sentences["snt_id"]))
sentence_lists = list(sentences["word"])

BATCH = 16

embs = {}

print("Computing sentence embeddings...")

for i in range(0, len(sentence_lists), BATCH):

    batch = sentence_lists[i:i+BATCH]

    enc = tok(
        batch,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc).last_hidden_state

    sent_embs = out.mean(dim=1).cpu().numpy()

    for j, key in enumerate(sentence_keys[i:i+BATCH]):
        embs[key] = sent_embs[j]

print(f"✅ computed embeddings for {len(embs)} sentences")

# ------------------------------------------------------------------
# Sentence boundaries
# ------------------------------------------------------------------

df["boundary"] = (
    (df["snt_id"] != df["snt_id"].shift(1)) |
    (df["run_id"] != df["run_id"].shift(1))
).astype(int)

df.loc[0, "boundary"] = 1

# ------------------------------------------------------------------
# Build stimulus-space regressors
# ------------------------------------------------------------------

stim_pre = {}
stim_hrf = {}

print("Building stimulus-space shift regressors...")

for run_id, g in df.groupby("run_id", sort=True):

    g = g.sort_values(["snt_id", "token_id"]).reset_index(drop=True)

    T = int(np.ceil(g["offset"].max() / TR)) + 1

    x = np.zeros(T, dtype=float)

    for i in range(1, len(g)):

        if g.iloc[i]["boundary"] != 1:
            continue

        prev_key = (
            int(g.iloc[i - 1]["run_id"]),
            int(g.iloc[i - 1]["snt_id"])
        )

        cur_key = (
            int(g.iloc[i]["run_id"]),
            int(g.iloc[i]["snt_id"])
        )

        if prev_key not in embs or cur_key not in embs:
            continue

        shift = cosine_distance(embs[prev_key], embs[cur_key])

        idx = int(round(float(g.iloc[i]["onset"]) / TR))

        if 0 <= idx < T:
            x[idx] = shift

    x_hrf = np.convolve(x, HRF)[:len(x)]

    stim_pre[run_id] = x
    stim_hrf[run_id] = x_hrf

print(f"✅ built regressors for stimulus runs: {sorted(stim_pre.keys())}")

# ------------------------------------------------------------------
# Map stimulus runs onto actual subject runs
# ------------------------------------------------------------------

subjects = sorted([p for p in PANG_OUT.glob("sub-*") if p.is_dir()])

stim_run_ids = sorted(stim_pre.keys())

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

        out_pre = PRE_DIR / f"{sub}_run-{subj_run:02d}.npy"
        out_hrf = HRF_DIR / f"{sub}_run-{subj_run:02d}.npy"

        np.save(out_pre, stim_pre[stim_run])
        np.save(out_hrf, stim_hrf[stim_run])

    print(f"✅ {sub}: wrote {n} sentence-shift regressors")

print("✅ done")