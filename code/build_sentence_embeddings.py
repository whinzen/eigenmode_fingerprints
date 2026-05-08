import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path.home() / "eigenmode_fingerprints"
CSV_PATH = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"
OUT_DIR = BASE / "pang_out" / "sentence_embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "lppEN_sentences.csv"
OUT_NPY = OUT_DIR / "lppEN_sentence_embeddings.npy"

# Load annotation
df = pd.read_csv(CSV_PATH)

required = ["word", "snt_id", "onset", "offset", "run_id"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Build sentence table
sent_df = (
    df.groupby("snt_id", sort=True)
      .agg(
          run_id=("run_id", "first"),
          onset=("onset", "first"),
          offset=("offset", "last"),
          text=("word", lambda x: " ".join(map(str, x)))
      )
      .reset_index()
)

# Sentence embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Compute embeddings
embeddings = model.encode(
    sent_df["text"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

# Save
sent_df.to_csv(OUT_CSV, index=False)
np.save(OUT_NPY, embeddings)

print("✅ Wrote:")
print(" ", OUT_CSV)
print(" ", OUT_NPY)
print("Embeddings shape:", embeddings.shape)