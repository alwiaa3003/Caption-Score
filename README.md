# CaptionScore

Predicts whether a short-video post is likely to land in the **top 10% of views for its month**,
using only its caption, hashtags and a few simple metadata features.

> **Status:** work in progress (Phase 1: data sampling). Sections marked _TBD_ are filled in as
> each phase is completed. No numbers appear here until they are computed and saved in
> `ml/reports/metrics.json`.

## Honest caveats (read first)
- The model is trained on **TikTok** data. Using it for Instagram Reels is an experiment, not a
  validated use.
- There is **no follower/creator column** in the data, so creator size, the biggest driver of
  views, is not controlled for. Captions can only explain a small part of virality.
- The dataset was collected through TikTok's private API against its terms of service. The app
  therefore **never displays raw captions from the dataset**.

## Data
- Source: Hugging Face [`datasocial/tiktok-5.6B-videos`](https://huggingface.co/datasets/datasocial/tiktok-5.6B-videos),
  licensed **CC BY-NC 4.0** (non-commercial use only). This project is non-commercial.
- A sample spread evenly over time (2022-01 to 2026-08, English, about 15k rows per month) is
  built by reading random Parquet row groups. See [docs/data_notes.md](docs/data_notes.md).

## Label definition
_TBD (Phase 2)._ See `docs/label_notes.md`.

## Results
_TBD (Phase 4)._ Will match `ml/reports/metrics.json` exactly.

## Repository layout
```
ml/             Python: data sampling, features, training, evaluation
model-service/  FastAPI service that loads the trained model
server/         Node + Express API (MongoDB via Mongoose)
client/         React (Vite) frontend
docs/           Data, label and design notes
```

## How to run
### ML (Python)
```bash
cd ml
python -m venv .venv
.venv/Scripts/activate        # Windows (use source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python -m src.inspect_schema  # print the schema + 5 example rows (streaming)
python -m src.sample_data     # build the time-spread sample into ml/data/
pytest
```
Each service will also run standalone (`uvicorn`, `npm run dev`). `docker-compose.yml` is
included for later use with Docker Desktop. _Details TBD._

## Limitations
_TBD._

## Ethics and license
_TBD (Phase 10)._
