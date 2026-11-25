# Model service (FastAPI)

Loads the final TF-IDF + LR bundle once at startup and serves predictions with exact per-word
explanations and hashtag suggestions.

| Endpoint | |
|---|---|
| `POST /predict` | `{caption, hashtags?, post_type?, duration_sec?}` -> `{probability, label, top_positive_tokens, top_negative_tokens, suggested_hashtags, model_version, base_rate, likely_threshold, tokens, notes}` |
| `GET /health` | `{status, model_version}` |
| `GET /model-info` | model card: label definition, validation/test metrics, caveats |
| `GET /docs` | interactive OpenAPI docs |

Validation: caption 1-2200 characters (not blank), at most 30 hashtags (letters, digits, `_`),
`post_type` in video/photo/reel/other, `0 < duration_sec <= 3600`. Invalid input -> 422.
`post_type` and `duration_sec` are accepted but not used by the text-only model (the response
says so in `notes`).

## Run locally (Windows paths; use `bin/` instead of `Scripts/` on macOS/Linux)
```bash
cd model-service
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/pip install --no-deps -e ../ml      # shared cleaning/explanation code
.venv/Scripts/uvicorn app.main:app --port 8000     # needs ml/models/final (or set MODEL_DIR)
.venv/Scripts/python -m pytest                     # tests use a tiny synthetic model
```

## Docker (not tested yet)
```bash
docker build -f model-service/Dockerfile -t captionscore-model .   # from the repo root
docker run -p 8000:8000 captionscore-model
```
