# Build from the REPO ROOT (it needs ml/src and ml/models/final):
#   docker build -f model-service/Dockerfile -t captionscore-model .
# Not tested yet. Without Docker, run the service with uvicorn (see README).
FROM python:3.13-slim

WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Dependencies first (cached layer). Versions are pinned to the training environment.
COPY model-service/requirements.txt model-service/requirements.txt
RUN pip install --no-cache-dir -r model-service/requirements.txt

# Shared ML code (text cleaning, prediction, explanations), installed as the `src` package
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/src ml/src
RUN pip install --no-cache-dir --no-deps ./ml

# Service code + the trained model bundle (built by `python -m src.build_explainability`)
COPY model-service/app model-service/app
COPY ml/models/final /srv/model
ENV MODEL_DIR=/srv/model

RUN useradd --create-home appuser
USER appuser
WORKDIR /srv/model-service
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
