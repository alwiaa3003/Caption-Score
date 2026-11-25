"""Explanations must add up exactly to the model's logit. Uses a small model trained here on
synthetic captions, so the test never needs the real dataset."""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.explain import CaptionModel, HashtagSuggester, build_text, display_piece
from src.train_baselines import WORD_TOKEN

CAPTIONS = [
    "wait for it 😂 #funny #fyp", "my morning routine #grwm", "link in bio for the recipe",
    "best pasta recipe ever 🍝 #food #recipe", "part 2 of the story time #storytime",
    "follow for more tips #tips", "cute dog doing tricks 🐶 #dog", "is this normal?? #fyp",
    "day 3 of learning guitar #guitar", "funny dog fail 😂 #dog #funny",
] * 5
LABELS = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1] * 5


@pytest.fixture(scope="module")
def model():
    from src.features import clean_text

    texts = [clean_text(c) for c in CAPTIONS]
    word = TfidfVectorizer(token_pattern=WORD_TOKEN, ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    x = sp.hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    lr = LogisticRegression(C=10, random_state=0).fit(x, LABELS)
    return CaptionModel({"word": word, "char": char, "lr": lr}), lr, word, char


@pytest.mark.parametrize("caption", [
    "funny dog 😂😂 #dog #funny", "the the the dog", "unseen words entirely zzz",
    "Check https://x.com @someone wait for it!!", "", "#fyp",
])
def test_contributions_sum_exactly_to_logit(model, caption):
    cm, lr, word, char = model
    exp = cm.explain(caption)
    from src.features import clean_text

    t = clean_text(caption)
    x = sp.hstack([word.transform([t]), char.transform([t])]).tocsr()
    expected = float(lr.decision_function(x)[0])
    assert exp.logit == pytest.approx(expected, abs=1e-9)
    assert exp.intercept + sum(exp.contributions) == pytest.approx(expected, abs=1e-9)
    assert exp.probability == pytest.approx(float(lr.predict_proba(x)[0, 1]), abs=1e-9)
    assert len(exp.pieces) == len(exp.contributions)


def test_predict_proba_matches_sklearn(model):
    cm, lr, word, char = model
    from src.features import clean_text

    texts = ["funny dog", "morning routine", "pasta"]
    x = sp.hstack([word.transform([clean_text(t) for t in texts]),
                   char.transform([clean_text(t) for t in texts])]).tocsr()
    np.testing.assert_allclose(cm.predict_proba(texts), lr.predict_proba(x)[:, 1], atol=1e-12)


def test_top_tokens_are_signed_and_sorted(model):
    cm, *_ = model
    pos, neg = cm.top_tokens(cm.explain("funny dog fail 😂 link in bio #dog"), k=3)
    assert all(t["weight"] > 0 for t in pos) and all(t["weight"] < 0 for t in neg)
    assert [t["weight"] for t in pos] == sorted([t["weight"] for t in pos], reverse=True)


def test_build_text_adds_only_new_valid_hashtags():
    assert build_text("hello #fyp", ["#fyp", "FOOD", "bad tag!", ""]) == "hello #fyp #food"


def test_display_piece():
    assert display_piece(":fire:") == "🔥"
    assert display_piece("xxurl") == "[link]"


def test_suggester_returns_only_unused_helpful_tags(model):
    cm, _, word, _ = model
    tags = ["dog", "funny", "recipe", "tips"]
    from src.features import clean_text

    rows = []
    for t in tags:
        docs = [clean_text(c) for c in CAPTIONS if f"#{t}" in c]
        m = word.transform(docs).mean(axis=0)
        rows.append(np.asarray(m).ravel() / (np.linalg.norm(m) or 1))
    sugg = HashtagSuggester({"tags": tags, "lift": np.array([2.0, 1.8, 1.2, 0.9]),
                             "centroids": sp.csr_matrix(np.vstack(rows))}, cm)
    out = sugg.suggest("cute dog doing a trick #dog", k=3)
    assert all(s["hashtag"] != "dog" for s in out)
    assert all(s["probability_change"] > 0 for s in out)


def test_suggester_returns_nothing_for_unrelated_caption(model):
    cm, _, word, _ = model
    row = np.asarray(word.transform(["guitar"]).todense()).ravel()
    sugg = HashtagSuggester({"tags": ["guitar"], "lift": np.array([2.0]),
                             "centroids": sp.csr_matrix(row / np.linalg.norm(row))}, cm)
    assert sugg.suggest("pasta recipe") == []
