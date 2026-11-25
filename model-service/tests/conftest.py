"""A tiny model bundle trained on synthetic captions, so the API tests never need the dataset
or the real (gitignored) model files."""

import json

import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from src.features import WORD_TOKEN, clean_text

CAPTIONS = ["wait for it 😂 #funny", "my morning routine #grwm", "link in bio for the recipe",
            "best pasta recipe ever #food", "funny dog fail 😂 #dog #funny",
            "cute dog tricks #dog"] * 5
LABELS = [1, 0, 0, 1, 1, 0] * 5


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    texts = [clean_text(c) for c in CAPTIONS]
    word = TfidfVectorizer(token_pattern=WORD_TOKEN, ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    x = sp.hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    lr = LogisticRegression(C=10, random_state=0).fit(x, LABELS)
    joblib.dump({"word": word, "char": char, "lr": lr}, d / "model.joblib")
    tags = ["funny", "dog", "food"]
    cents = np.vstack([np.asarray(word.transform([clean_text(c) for c in CAPTIONS
                                                  if f"#{t}" in c]).mean(axis=0)).ravel()
                       for t in tags])
    joblib.dump({"tags": tags, "lift": np.array([2.0, 1.5, 1.2]),
                 "centroids": normalize(sp.csr_matrix(cents))}, d / "suggest.joblib")
    (d / "model_card.json").write_text(json.dumps({
        "model_version": "test-model", "base_rate": 0.1,
        "label_cutoff": {"likely_threshold": 0.5}}))
    return d


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from app.main import app

    with TestClient(app) as c:  # entering runs the lifespan (loads the model once)
        yield c
