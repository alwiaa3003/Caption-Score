"""Prediction, per-word explanations and hashtag suggestions for the final TF-IDF + LR model.

Used by the FastAPI model service, so it must stay importable without pandas-heavy work.

Per-word explanations are EXACT, not approximated (no SHAP/LIME sampling needed):
the model is linear, logit = intercept + sum_j x_j * w_j over TF-IDF features j.
Every feature is assigned to the caption "pieces" (whitespace-separated tokens of the
cleaned text) it came from:
  - word unigram  -> the piece(s) containing that token (split equally across occurrences)
  - word bigram   -> half to each of the two pieces (per occurrence)
  - char n-gram   -> the piece(s) containing it; `char_wb` n-grams never cross whitespace,
                     so each one lies inside exactly one piece
So the piece contributions add up exactly to logit - intercept (tested).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import emoji
import numpy as np
import scipy.sparse as sp

from src.features import clean_text, extract_hashtags

MAX_CAPTION_CHARS = 2200  # TikTok's own caption limit; the service rejects longer input


def build_text(caption: str, hashtags: list[str] | None = None) -> str:
    """Caption plus any extra hashtags not already in it, as one string."""
    present = set(extract_hashtags(caption))
    extra = []
    for h in hashtags or []:
        tag = h.strip().lstrip("#").lower()
        if tag and re.fullmatch(r"\w+", tag) and tag not in present:
            extra.append("#" + tag)
            present.add(tag)
    return (caption + " " + " ".join(extra)).strip()


def display_piece(piece: str) -> str:
    """Turn a cleaned piece back into something readable (':fire:' -> the emoji)."""
    shown = emoji.emojize(piece, language="alias") if ":" in piece else piece
    return {"xxurl": "[link]", "xxmention": "[@mention]"}.get(shown, shown)


@dataclass
class Explanation:
    probability: float
    logit: float
    intercept: float
    pieces: list[str]            # cleaned text split on whitespace
    contributions: list[float]   # one per piece, in order


class CaptionModel:
    """Wraps the saved bundle {word, char, lr} from ml/models/final/model.joblib."""

    def __init__(self, bundle: dict):
        self.word = bundle["word"]
        self.char = bundle["char"]
        coef = bundle["lr"].coef_.ravel()
        self.n_word = len(self.word.vocabulary_)
        self.w_word, self.w_char = coef[:self.n_word], coef[self.n_word:]
        self.intercept = float(bundle["lr"].intercept_[0])
        self._word_tok = re.compile(self.word.token_pattern)
        self._char_an = self.char.build_analyzer()
        self._word_names = self.word.get_feature_names_out()
        self._char_names = self.char.get_feature_names_out()

    # ---- scoring ---------------------------------------------------------------------------
    def logits(self, texts: list[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        xw, xc = self.word.transform(cleaned), self.char.transform(cleaned)
        return xw @ self.w_word + xc @ self.w_char + self.intercept

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits(texts)))

    # ---- explanation -----------------------------------------------------------------------
    def explain(self, text: str) -> Explanation:
        cleaned = clean_text(text)
        pieces = cleaned.split()
        contrib = np.zeros(len(pieces))
        if pieces:
            self._add_word_contributions(cleaned, pieces, contrib)
            self._add_char_contributions(cleaned, pieces, contrib)
        logit = self.intercept + float(contrib.sum())
        return Explanation(probability=1.0 / (1.0 + np.exp(-logit)), logit=logit,
                           intercept=self.intercept, pieces=pieces,
                           contributions=contrib.tolist())

    def _add_word_contributions(self, cleaned, pieces, contrib) -> None:
        # piece index for every character offset of `cleaned` (pieces are single-space joined)
        owner = np.full(len(cleaned), -1)
        pos = 0
        for i, p in enumerate(pieces):
            start = cleaned.index(p, pos)
            owner[start:start + len(p)] = i
            pos = start + len(p)
        toks = [(m.group(), owner[m.start()]) for m in self._word_tok.finditer(cleaned)]
        x = self.word.transform([cleaned])
        for j, val in zip(x.indices, x.data, strict=True):
            c = val * self.w_word[j]
            gram = self._word_names[j].split(" ")
            hits = [i for i in range(len(toks) - len(gram) + 1)
                    if [t for t, _ in toks[i:i + len(gram)]] == gram]
            for i in hits:
                for t in range(len(gram)):
                    contrib[toks[i + t][1]] += c / len(hits) / len(gram)

    def _add_char_contributions(self, cleaned, pieces, contrib) -> None:
        x = self.char.transform([cleaned])
        if x.nnz == 0:
            return
        grams_per_piece = [self._char_an(p) for p in pieces]
        for j, val in zip(x.indices, x.data, strict=True):
            g = self._char_names[j]
            counts = np.array([grams.count(g) for grams in grams_per_piece], dtype=float)
            contrib += val * self.w_char[j] * counts / counts.sum()

    def top_tokens(self, exp: Explanation, k: int = 5) -> tuple[list[dict], list[dict]]:
        """Aggregate identical pieces; return (top positive, top negative) by contribution."""
        agg: dict[str, float] = {}
        for p, c in zip(exp.pieces, exp.contributions, strict=True):
            agg[p] = agg.get(p, 0.0) + c
        items = [{"token": display_piece(p), "weight": round(c, 4)} for p, c in agg.items()]
        pos = sorted((t for t in items if t["weight"] > 0), key=lambda t: -t["weight"])[:k]
        neg = sorted((t for t in items if t["weight"] < 0), key=lambda t: t["weight"])[:k]
        return pos, neg


class HashtagSuggester:
    """Suggest hashtags with a higher top-10% rate that fit the caption.

    1. candidates: hashtags seen in >= 500 training posts whose viral-rate Wilson lower bound
       is above the base rate (hashtag_lift table)
    2. relevance: cosine between the caption's word TF-IDF vector and each candidate's
       centroid (mean word TF-IDF of training captions using it)
    3. keep the `pool` most relevant candidates with relevance >= MIN_RELEVANCE, drop any whose
       addition doesn't raise the model's probability, and rank the rest by that increase.
       If nothing is relevant enough, return [] (better no suggestion than an unrelated one).
    Association, not causation: these tags co-occur with higher rates; adding one is not
    guaranteed to help.
    """

    def __init__(self, bundle: dict, model: CaptionModel):
        self.tags: list[str] = bundle["tags"]
        self.lift: np.ndarray = bundle["lift"]
        self.centroids: sp.csr_matrix = bundle["centroids"]  # rows L2-normalised
        self.model = model

    MIN_RELEVANCE = 0.05

    def suggest(self, text: str, k: int = 5, pool: int = 10) -> list[dict]:
        present = set(extract_hashtags(text))
        q = self.model.word.transform([clean_text(text)])
        rel = np.asarray((self.centroids @ q.T).todense()).ravel()
        order = np.argsort(-rel, kind="stable")
        cand = [i for i in order
                if self.tags[i] not in present and rel[i] >= self.MIN_RELEVANCE][:pool]
        if not cand:
            return []
        base = float(self.model.predict_proba([text])[0])
        probs = self.model.predict_proba([f"{text} #{self.tags[i]}" for i in cand])
        out = [{"hashtag": self.tags[i], "lift": round(float(self.lift[i]), 2),
                "relevance": round(float(rel[i]), 3),
                "probability_change": round(float(p - base), 4)}
               for i, p in zip(cand, probs, strict=True) if p > base]
        return sorted(out, key=lambda d: -d["probability_change"])[:k]
