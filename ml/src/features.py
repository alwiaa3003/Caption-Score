"""Text cleaning and meta features. The same functions are used for training and serving.

Deliberately NOT features:
  - posting hour / day of week: each sampled month covers only ~2-4 days, so these would
    encode which days got sampled, not real timing effects (docs/data_notes.md)
  - country / language: v1 is US + English only, so they are constant
  - views, stats_updated_at, age at capture, likes...: unknown at posting time / leak the label
"""

import re

import emoji
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[\w.]+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")
# TF-IDF word tokens: emoji names like ":fire:", hashtags "#fyp", and ordinary words.
WORD_TOKEN = r"(?u):[\w&'-]+:|#?\w+"

# Call-to-action phrases. Word boundaries stop "follower" matching "follow" etc.
CTA_RE = re.compile(
    r"\b(link in (my )?bio|follow( me| for)?|comment|like (and|&) share|share (this|with)"
    r"|subscribe|tag (a|your) (friend|bestie)|save (this|for later)|duet|stitch)\b",
    re.IGNORECASE,
)


def extract_hashtags(caption: str) -> list[str]:
    """Lower-case hashtags without '#', in order, de-duplicated."""
    seen: dict[str, None] = {}
    for tag in HASHTAG_RE.findall(caption or ""):
        seen.setdefault(tag.lower(), None)
    return list(seen)


def clean_text(caption: str) -> str:
    """Lower-case; URLs -> 'xxurl'; @mentions -> 'xxmention'; emojis -> ' :emoji_name: '.

    Hashtags stay in the text as '#tag' tokens, so text models can use them.
    Placeholder tokens are plain words so TF-IDF tokenizers keep them.
    """
    text = caption or ""
    text = URL_RE.sub(" xxurl ", text)
    text = MENTION_RE.sub(" xxmention ", text)
    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = text.lower()
    return WHITESPACE_RE.sub(" ", text).strip()


def meta_features(caption: str, duration_sec: float | None) -> dict[str, float]:
    """Simple numeric features from the raw caption and video duration."""
    caption = caption or ""
    no_tags = HASHTAG_RE.sub("", caption)
    letters = [c for c in no_tags if c.isalpha()]
    return {
        "caption_chars": len(caption),
        "word_count": len(no_tags.split()),
        "hashtag_count": len(HASHTAG_RE.findall(caption)),
        "emoji_count": emoji.emoji_count(caption),
        "mention_count": len(MENTION_RE.findall(caption)),
        "has_url": float(bool(URL_RE.search(caption))),
        "has_question": float("?" in caption),
        "has_exclamation": float("!" in caption),
        "has_cta": float(bool(CTA_RE.search(caption))),
        "upper_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "duration_sec": float(duration_sec) if duration_sec is not None else float("nan"),
    }


META_COLUMNS = list(meta_features("", 0.0).keys())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `text`, `hashtags` and META_COLUMNS to a frame with `caption` and `duration_ms`."""
    out = df.copy()
    out["text"] = out["caption"].map(clean_text)
    out["hashtags"] = out["caption"].map(extract_hashtags)
    meta = [meta_features(c, d / 1000.0) for c, d in zip(out["caption"], out["duration_ms"],
                                                           strict=True)]
    return pd.concat([out, pd.DataFrame(meta, index=out.index)], axis=1)
