"""CaptionScore model service (FastAPI).

POST /predict      score a caption, explain it, suggest hashtags
GET  /health       liveness + loaded model version
GET  /model-info   model card (metrics, label definition, caveats) for the About page

The model is loaded ONCE at startup (lifespan) from MODEL_DIR, which must contain
model.joblib, suggest.joblib and model_card.json (built by `python -m src.build_explainability`).

Run:  uvicorn app.main:app --port 8000
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from src.explain import CaptionModel, HashtagSuggester, build_text, display_piece

log = logging.getLogger("model-service")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "ml" / "models" / "final"


class Loaded:
    model: CaptionModel
    suggester: HashtagSuggester
    card: dict


def load_bundle(model_dir: Path) -> Loaded:
    missing = [f for f in ("model.joblib", "suggest.joblib", "model_card.json")
               if not (model_dir / f).exists()]
    if missing:
        raise RuntimeError(f"model files missing in {model_dir}: {missing}. "
                           "Run `python -m src.build_explainability` in ml/ first.")
    loaded = Loaded()
    loaded.model = CaptionModel(joblib.load(model_dir / "model.joblib"))
    loaded.suggester = HashtagSuggester(joblib.load(model_dir / "suggest.joblib"), loaded.model)
    loaded.card = json.loads((model_dir / "model_card.json").read_text())
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    app.state.loaded = load_bundle(model_dir)  # fail fast: no model, no service
    log.info("loaded model %s from %s", app.state.loaded.card["model_version"], model_dir)
    yield


app = FastAPI(title="CaptionScore model service", version="1.0.0", lifespan=lifespan)


def get_loaded() -> Loaded:
    loaded = getattr(app.state, "loaded", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return loaded


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=get_loaded().card["model_version"])


@app.get("/model-info")
def model_info() -> dict:
    return get_loaded().card


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    lo = get_loaded()
    text = build_text(req.caption, req.hashtags)
    exp = lo.model.explain(text)
    pos, neg = lo.model.top_tokens(exp, k=5)
    threshold = lo.card["label_cutoff"]["likely_threshold"]
    notes = ["Trained on TikTok (US, English) captions; other platforms are an experiment.",
             "Probability of landing in the top 10% of comparable posts, not a view count."]
    if req.duration_sec is not None or req.post_type is not None:
        notes.append("post_type and duration_sec are accepted but not used by this "
                     "text-only model.")
    return PredictResponse(
        probability=round(exp.probability, 4),
        label="likely" if exp.probability >= threshold else "unlikely",
        top_positive_tokens=pos,
        top_negative_tokens=neg,
        suggested_hashtags=lo.suggester.suggest(text, k=5),
        model_version=lo.card["model_version"],
        base_rate=lo.card["base_rate"],
        likely_threshold=threshold,
        tokens=[{"token": display_piece(p), "weight": round(c, 4)}
                for p, c in zip(exp.pieces, exp.contributions, strict=True)],
        notes=notes,
    )
