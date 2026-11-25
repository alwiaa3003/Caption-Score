# CaptionScore

Predicts whether a short-video post is likely to land in the **top 10% of views among comparable
posts from its month**,
using only its caption, hashtags and a few simple metadata features.

> **Status:** work in progress (Phase 2: labels, splits and features). Sections marked _TBD_ are filled in as
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
- A sample spread evenly over time (2022-01 to 2026-08, US, English, 15k rows per month) is
  built by reading random Parquet row groups. See [docs/data_notes.md](docs/data_notes.md).

## Label definition
"Viral" = top 10% of views among US English videos posted in the same month **and** in the
same age-at-capture tertile of that month (ads excluded). Details and weaknesses:
[docs/label_notes.md](docs/label_notes.md).

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
python -m src.sample_data --dry-run  # footers only: shows the download size
python -m src.sample_data --confirm  # build the time-spread sample into ml/data/ (~4.95 GB streamed)
python -m src.describe_sample        # profile numbers used in docs/data_notes.md
python -m src.resample_us --confirm  # step 2: replace mixed-country row groups -> US-only (~0.75 GB)
python -m src.analyze_labels        # evidence for the label definition
python -m src.build_dataset         # labels + time splits + features -> ml/data/dataset.parquet
python -m src.dedup_check           # exact / near-duplicate captions across splits
python -m src.train_baselines       # Phase 3 baselines on validation -> reports/metrics.json
python -m src.plot_baselines        # reports/val_per_month_pr_auc.png
python -m src.embed --splits train val  # MiniLM caption embeddings (~60 min on CPU, resumable)
python -m src.train_embeddings      # Phase 4a: LR / LightGBM on embeddings
python -m src.compare_models        # validation comparison table -> reports/model_comparison.md
python -m src.select_model          # choose final model on validation (+ bootstrap, PR curves)
python -m src.evaluate_test         # ONE-TIME test + stress evaluation of the chosen model
python -m src.build_explainability  # Phase 5: explanations, hashtag lift, exports ml/models/final/
python -m src.export_colab          # Phase 4b: data for Colab -> ml/data/colab/
#   then run ml/notebooks/finetune_distilbert_colab.ipynb on a Colab T4 GPU
python -m src.import_transformer_preds  # score the DistilBERT predictions brought back
pytest
```
Each service will also run standalone (`uvicorn`, `npm run dev`). `docker-compose.yml` is
included for later use with Docker Desktop. _Details TBD._

## Limitations
_TBD._

## Ethics and license
_TBD (Phase 10)._
