import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.explain import MAX_CAPTION_CHARS

REAL_MODEL = Path(__file__).resolve().parents[2] / "ml" / "models" / "final"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "test-model"}


def test_predict_returns_contract(client):
    r = client.post("/predict", json={"caption": "funny dog 😂", "hashtags": ["#dog", "pets"],
                                      "post_type": "video", "duration_sec": 15})
    assert r.status_code == 200
    body = r.json()
    for key in ["probability", "label", "top_positive_tokens", "top_negative_tokens",
                "suggested_hashtags", "model_version"]:
        assert key in body
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in {"likely", "unlikely"}
    assert body["label"] == ("likely" if body["probability"] >= 0.5 else "unlikely")
    assert all(t["weight"] > 0 for t in body["top_positive_tokens"])
    assert all(t["weight"] < 0 for t in body["top_negative_tokens"])
    assert body["model_version"] == "test-model"
    # extra hashtags were appended; the one already present is not suggested again
    assert "#pets" in [t["token"] for t in body["tokens"]]
    assert "dog" not in [s["hashtag"] for s in body["suggested_hashtags"]]
    assert any("not used" in n for n in body["notes"])


def test_token_contributions_sum_to_logit(client, model_dir):
    import math

    import joblib

    intercept = float(joblib.load(model_dir / "model.joblib")["lr"].intercept_[0])
    body = client.post("/predict", json={"caption": "best pasta recipe #food 😂"}).json()
    logit = intercept + sum(t["weight"] for t in body["tokens"])
    # weights are rounded to 4 dp per token in the response
    assert 1 / (1 + math.exp(-logit)) == pytest.approx(body["probability"], abs=1e-3)


@pytest.mark.parametrize("payload,fragment", [
    ({}, "caption"),
    ({"caption": "   "}, "empty"),
    ({"caption": "x" * (MAX_CAPTION_CHARS + 1)}, "caption"),
    ({"caption": "ok", "hashtags": ["bad tag!"]}, "invalid hashtag"),
    ({"caption": "ok", "hashtags": ["a"] * 31}, "hashtags"),
    ({"caption": "ok", "post_type": "story"}, "post_type"),
    ({"caption": "ok", "duration_sec": -5}, "duration_sec"),
    ({"caption": 123}, "caption"),
])
def test_invalid_input_is_rejected(client, payload, fragment):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert fragment in r.text


def test_caption_at_limit_is_accepted(client):
    r = client.post("/predict", json={"caption": "a " * (MAX_CAPTION_CHARS // 2)})
    assert r.status_code == 200


def test_missing_model_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from app.main import app

    with pytest.raises(RuntimeError, match="model files missing"), TestClient(app):
        pass


@pytest.mark.skipif(not (REAL_MODEL / "model.joblib").exists(),
                    reason="real model not built (python -m src.build_explainability)")
def test_real_model_smoke(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(REAL_MODEL))
    from app.main import app

    with TestClient(app) as c:
        r = c.post("/predict", json={"caption": "wait for the end 😂😂 #funny #comedy"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_version"] == "tfidf-lr-v1"
        assert 0 < body["probability"] < 1
        assert os.environ["MODEL_DIR"] == str(REAL_MODEL)
