"""Encode cleaned captions with a sentence-transformer, cached in resumable chunks.

Timing on this machine (CPU, 6 threads, max_seq_length=64, batch 64): ~213 captions/s,
so all 840k rows would take ~66 min. Phase 4a encodes train + val only (~60 min);
test/stress are encoded only for the final evaluation.

Each chunk is saved as data/emb_<model>/<split>_<chunk>.npz with `video_id` and `emb`
(float16 to halve disk use). Existing chunks are skipped, so an interrupted run resumes.

Usage:  python -m src.embed --splits train val
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK = 20_000


def emb_dir(model: str = MODEL) -> Path:
    return REPO_ROOT / "ml" / "data" / f"emb_{model.split('/')[-1]}"


def load_embeddings(split: str, model: str = MODEL) -> tuple[np.ndarray, np.ndarray]:
    """Return (video_ids, float32 embeddings) for one split, in dataset row order."""
    files = sorted(emb_dir(model).glob(f"{split}_*.npz"))
    if not files:
        raise FileNotFoundError(f"no embeddings for {split}; run python -m src.embed")
    parts = [np.load(f) for f in files]
    return (np.concatenate([p["video_id"] for p in parts]),
            np.concatenate([p["emb"] for p in parts]).astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer  # heavy import, only when encoding

    torch.manual_seed(42)
    model = SentenceTransformer(MODEL, device="cpu")
    model.max_seq_length = 64
    out_dir = emb_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        df = pd.read_parquet(REPO_ROOT / "ml/data/dataset.parquet", columns=["video_id", "text"],
                             filters=[("split", "==", split)])
        n_chunks = (len(df) + CHUNK - 1) // CHUNK
        for i in range(n_chunks):
            path = out_dir / f"{split}_{i:04d}.npz"
            if path.exists():
                continue
            part = df.iloc[i * CHUNK:(i + 1) * CHUNK]
            t0 = time.time()
            emb = model.encode(part["text"].tolist(), batch_size=args.batch_size,
                               show_progress_bar=False, convert_to_numpy=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, video_id=part["video_id"].to_numpy(), emb=emb.astype(np.float16))
            tmp.replace(path)  # atomic: a half-written chunk never looks finished
            print(f"[{time.strftime('%H:%M:%S')}] {split} chunk {i + 1}/{n_chunks}: "
                  f"{len(part):,} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
