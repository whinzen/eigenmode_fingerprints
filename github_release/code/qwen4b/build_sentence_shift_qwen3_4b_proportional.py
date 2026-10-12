#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

QWEN_FILE = (
    PANG
    / "token_embeddings_qwen3_4b_proportional"
    / "token_embeddings_qwen3_4b_continuous_layer_m5.pkl"
)

OUT_DIR = (
    PANG
    / "sentence_level_controls_qwen3_4b_proportional"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FILE = (
    OUT_DIR
    / "sentence_shift_qwen3_4b_continuous.csv"
)


def cosine_distance(a, b):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return np.nan

    cos = np.dot(a, b) / (na * nb)
    cos = np.clip(cos, -1.0, 1.0)

    return 1.0 - cos


qwen = pd.read_pickle(QWEN_FILE)

required = [
    "word",
    "snt_id",
    "token_id",
    "onset",
    "offset",
    "run_id",
    "embedding",
]

missing = [c for c in required if c not in qwen.columns]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

qwen = (
    qwen
    .sort_values(
        ["run_id", "snt_id", "token_id"]
    )
    .reset_index(drop=True)
)

print("Qwen word rows:", len(qwen))
print("Stimulus runs:",
      sorted(qwen["run_id"].unique().tolist()))


# ------------------------------------------------------------
# Mean-pool ORIGINAL continuous-context word embeddings
# within each sentence.
#
# Deliberately no word-level L2 normalization here:
# this reproduces the v3 sentence-shift definition.
# ------------------------------------------------------------

sentence_rows = []

for (run_id, snt_id), g in qwen.groupby(
    ["run_id", "snt_id"],
    sort=True,
):

    g = g.sort_values("token_id")

    embs = np.stack(
        g["embedding"]
        .apply(
            lambda x: np.asarray(
                x,
                dtype=np.float64,
            )
        )
        .to_list(),
        axis=0,
    )

    sentence_embedding = embs.mean(axis=0)

    sentence_rows.append({
        "run_id": int(run_id),
        "snt_id": int(snt_id),
        "sentence_onset": float(
            g["onset"].iloc[0]
        ),
        "sentence_offset": float(
            g["offset"].iloc[-1]
        ),
        "n_words": int(len(g)),
        "sentence_embedding": sentence_embedding,
    })

sent = pd.DataFrame(sentence_rows)

print("\nNumber of sentences:", len(sent))
print("Sentences by run:")
print(sent.groupby("run_id").size())


# ------------------------------------------------------------
# Consecutive-sentence cosine distance within run.
# Shift belongs to INCOMING sentence.
# ------------------------------------------------------------

shift_rows = []

for run_id, g in sent.groupby(
    "run_id",
    sort=True,
):

    g = (
        g.sort_values("snt_id")
         .reset_index(drop=True)
    )

    for i in range(1, len(g)):

        shift = cosine_distance(
            g.loc[i - 1, "sentence_embedding"],
            g.loc[i, "sentence_embedding"],
        )

        shift_rows.append({
            "run_id": int(run_id),
            "previous_snt_id": int(
                g.loc[i - 1, "snt_id"]
            ),
            "snt_id": int(
                g.loc[i, "snt_id"]
            ),
            "onset": float(
                g.loc[i, "sentence_onset"]
            ),
            "n_words": int(
                g.loc[i, "n_words"]
            ),
            "sentence_shift": float(shift),
        })

shift_df = pd.DataFrame(shift_rows)


# ------------------------------------------------------------
# Exact v3 convention: center WITHIN stimulus run.
# ------------------------------------------------------------

shift_df["sentence_shift_centered"] = (
    shift_df["sentence_shift"]
    -
    shift_df.groupby("run_id")[
        "sentence_shift"
    ].transform("mean")
)

shift_df.to_csv(EVENT_FILE, index=False)

print("\n========================================")
print("SENTENCE-SHIFT QC")
print("========================================")
print("Output:", EVENT_FILE)
print("Transitions:", len(shift_df))
print(
    "Expected:",
    len(sent) - sent["run_id"].nunique()
)

print("\nRaw sentence shift:")
print(shift_df["sentence_shift"].describe())

print("\nCentered sentence shift:")
print(
    shift_df[
        "sentence_shift_centered"
    ].describe()
)

print("\nCentered mean by run:")
print(
    shift_df.groupby("run_id")[
        "sentence_shift_centered"
    ].mean()
)

assert not shift_df[
    "sentence_shift"
].isna().any()

assert np.allclose(
    shift_df.groupby("run_id")[
        "sentence_shift_centered"
    ].mean().values,
    0.0,
    atol=1e-12,
)

print("\nQC PASSED")
