"""Leakage check: exact and near-duplicate captions across train / val / test / stress.

Reposts and templated captions ("Part 2 #fyp", giveaway templates...) can appear in both train
and val. A model that memorises them looks better than it is. This script counts them and
writes, for every row, flags saying whether its caption also occurs in train:

  dup_exact_in_train  cleaned text (src.features.clean_text) identical to some train row
  dup_near_in_train   word-3-shingle Jaccard >= 0.8 with some train row (MinHash + LSH),
                      or identical after a looser normalisation (digits -> 0, punctuation
                      and emoji names removed, hashtags sorted)

Only captions are compared; no labels or scores are used, so checking test here doesn't
"touch" the test set in the evaluation sense.

Usage:  python -m src.dedup_check
Writes: reports/dedup_report.json, data/dedup_flags.parquet
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent

N_PERM = 64          # MinHash permutations
BANDS, ROWS = 8, 8   # LSH: 8 bands x 8 rows -> ~0.77 Jaccard detection threshold
JACCARD = 0.8
MAX_CANDIDATES = 50  # verify at most this many train candidates per query row
_PRIME = (1 << 61) - 1

_EMOJI_NAME = re.compile(r":[a-z0-9_&'-]+:")
_PUNCT = re.compile(r"[^\w#\s]")
_HASHTAG = re.compile(r"#\w+")


def loose_key(text: str) -> str:
    """Looser normalisation for templated captions: digits -> 0, no punctuation/emoji names,
    hashtags moved to the end in sorted order."""
    t = _EMOJI_NAME.sub(" ", text)
    tags = sorted(set(_HASHTAG.findall(t)))
    t = _HASHTAG.sub(" ", t)
    t = re.sub(r"\d", "0", _PUNCT.sub(" ", t))
    return " ".join(t.split() + tags)


def shingles(text: str) -> set[str]:
    words = text.split()
    if len(words) < 3:
        return {" ".join(words)}
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def minhash_signatures(docs: list[set[str]], seed: int = 0) -> np.ndarray:
    """(n_docs, N_PERM) MinHash signatures with universal hashing a*x+b mod p."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, N_PERM, dtype=np.uint64).astype(object)
    b = rng.integers(0, _PRIME, N_PERM, dtype=np.uint64).astype(object)
    a = np.array(a % (1 << 31), dtype=np.uint64) | np.uint64(1)
    b = np.array(b % (1 << 31), dtype=np.uint64)
    sig = np.empty((len(docs), N_PERM), dtype=np.uint64)
    for i, sh in enumerate(docs):
        # stable 32-bit hashes of each shingle (Python's hash() is salted per process,
        # which is fine here because all signatures are computed in this one process)
        x = np.fromiter((hash(s) & 0xFFFFFFFF for s in sh), dtype=np.uint64, count=len(sh))
        sig[i] = ((np.outer(x, a) + b) % np.uint64((1 << 31) - 1)).min(axis=0)
    return sig


def lsh_near_dups(query_sig, query_sh, ref_sig, ref_sh) -> np.ndarray:
    """For each query doc: does some ref doc have Jaccard >= JACCARD? (bool array)"""
    buckets: dict[tuple, list[int]] = {}
    for j in range(len(ref_sig)):
        for bnd in range(BANDS):
            key = (bnd, ref_sig[j, bnd * ROWS:(bnd + 1) * ROWS].tobytes())
            buckets.setdefault(key, []).append(j)
    found = np.zeros(len(query_sig), dtype=bool)
    for i in range(len(query_sig)):
        cands: list[int] = []
        for bnd in range(BANDS):
            key = (bnd, query_sig[i, bnd * ROWS:(bnd + 1) * ROWS].tobytes())
            cands.extend(buckets.get(key, ())[:MAX_CANDIDATES])
            if len(cands) >= MAX_CANDIDATES:
                break
        q = query_sh[i]
        for j in dict.fromkeys(cands):
            r = ref_sh[j]
            if len(q & r) / len(q | r) >= JACCARD:
                found[i] = True
                break
    return found


def main() -> None:
    df = pd.read_parquet(REPO_ROOT / "ml/data/dataset.parquet",
                         columns=["video_id", "split", "text", "viral"])
    df["loose"] = df["text"].map(loose_key)
    train = df[df["split"] == "train"]
    train_exact, train_loose = set(train["text"]), set(train["loose"])
    df["dup_exact_in_train"] = df["text"].isin(train_exact) & (df["split"] != "train")
    df["dup_loose_in_train"] = df["loose"].isin(train_loose) & (df["split"] != "train")

    # MinHash near-dups for non-train rows vs train (skip rows already flagged loose/exact)
    print("computing MinHash signatures...", flush=True)
    sh = df["text"].map(shingles).tolist()
    sig = minhash_signatures(sh)
    is_train = (df["split"] == "train").to_numpy()
    ref_idx = np.flatnonzero(is_train)
    q_idx = np.flatnonzero(~is_train & ~df["dup_loose_in_train"].to_numpy())
    print(f"LSH: {len(q_idx):,} query rows vs {len(ref_idx):,} train rows", flush=True)
    near = lsh_near_dups(sig[q_idx], [sh[i] for i in q_idx], sig[ref_idx],
                         [sh[i] for i in ref_idx])
    df["dup_near_in_train"] = df["dup_loose_in_train"]
    df.loc[df.index[q_idx[near]], "dup_near_in_train"] = True

    report: dict = {"method": {"near": f"word-3-shingle Jaccard >= {JACCARD} via MinHash "
                                       f"({N_PERM} perms, {BANDS}x{ROWS} LSH) OR loose-key match",
                               "exact": "identical cleaned text"}}
    for split in ["train", "val", "test", "stress"]:
        s = df[df["split"] == split]
        entry = {
            "rows": len(s),
            "exact_dup_within_split_rows": int(s["text"].duplicated(keep=False).sum()),
            "exact_dup_within_split_share": round(float(s["text"].duplicated(keep=False)
                                                        .mean()), 4),
        }
        if split != "train":
            for col in ["dup_exact_in_train", "dup_near_in_train"]:
                entry[f"{col}_rows"] = int(s[col].sum())
                entry[f"{col}_share"] = round(float(s[col].mean()), 4)
            # Do duplicated captions carry the label with them? (val/test labels are not used
            # for any model decision here; this is just descriptive)
            if split == "val":
                entry["viral_rate_if_near_dup"] = round(float(s.loc[s["dup_near_in_train"],
                                                                    "viral"].mean()), 4)
                entry["viral_rate_if_not_dup"] = round(float(s.loc[~s["dup_near_in_train"],
                                                                   "viral"].mean()), 4)
        report[split] = entry
    top = df[df["split"] == "val"]
    top = top[top["dup_exact_in_train"]]["text"].str.len().describe().round(1).to_dict()
    report["val_exact_dup_caption_chars"] = top

    (ML_DIR / "reports" / "dedup_report.json").write_text(json.dumps(report, indent=2))
    df[["video_id", "split", "dup_exact_in_train", "dup_near_in_train"]].to_parquet(
        REPO_ROOT / "ml/data/dedup_flags.parquet", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
