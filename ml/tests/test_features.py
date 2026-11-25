import math

import pandas as pd

from src.features import META_COLUMNS, build_features, clean_text, extract_hashtags, meta_features


def test_extract_hashtags_lowercases_and_dedupes():
    assert extract_hashtags("Hi #FYP #fyp #Cooking_Tips!") == ["fyp", "cooking_tips"]
    assert extract_hashtags("") == []
    assert extract_hashtags(None) == []


def test_clean_text_normalizes_urls_mentions_and_emojis():
    t = clean_text("Check https://x.com/a NOW @some.user 🔥 #Food")
    assert "xxurl" in t and "https" not in t
    assert "xxmention" in t and "some.user" not in t
    assert ":fire:" in t
    assert "#food" in t
    assert t == t.lower()


def test_meta_features_counts():
    f = meta_features("Wait for it?! link in bio 😂😂 #fun #lol @pal", 12.5)
    assert f["hashtag_count"] == 2
    assert f["emoji_count"] == 2
    assert f["mention_count"] == 1
    assert f["has_question"] == 1.0 and f["has_exclamation"] == 1.0
    assert f["has_cta"] == 1.0
    assert f["duration_sec"] == 12.5


def test_cta_uses_word_boundaries():
    assert meta_features("my followers are great", 1)["has_cta"] == 0.0
    assert meta_features("follow for part 2", 1)["has_cta"] == 1.0


def test_empty_caption_is_safe():
    f = meta_features("", None)
    assert f["caption_chars"] == 0 and f["upper_ratio"] == 0.0
    assert math.isnan(f["duration_sec"])


def test_build_features_adds_all_columns():
    df = pd.DataFrame({"caption": ["Hello #a", "World"], "duration_ms": [1000, 2500]})
    out = build_features(df)
    for col in ["text", "hashtags", *META_COLUMNS]:
        assert col in out
    assert out["duration_sec"].tolist() == [1.0, 2.5]
    assert out["hashtags"].tolist() == [["a"], []]


def test_no_timing_features():
    # posting hour / weekday were removed on purpose (sampling covers only a few days a month)
    assert not any(c in META_COLUMNS for c in ("hour", "posting_hour", "day_of_week"))
