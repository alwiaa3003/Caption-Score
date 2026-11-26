"""Download the trained model bundle for a deployment build.

The model files (ml/models/final) are not in git. For hosting, they are attached to a GitHub
Release as one zip. This script downloads that zip, checks its SHA-256 and unpacks it into
ml/models/final, which is where the service looks by default.

    MODEL_BUNDLE_URL     URL of the zip (required)
    MODEL_BUNDLE_SHA256  expected SHA-256 of the zip (optional, checked when set)

Run from the repo root:  python model-service/fetch_model.py
"""

import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "ml" / "models" / "final"
FILES = {"model.joblib", "suggest.joblib", "model_card.json"}


def main() -> None:
    url = os.environ.get("MODEL_BUNDLE_URL", "").strip()
    if not url:
        sys.exit("MODEL_BUNDLE_URL is not set.")

    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    expected = os.environ.get("MODEL_BUNDLE_SHA256", "").strip().lower()
    if expected and digest != expected:
        sys.exit(f"Checksum mismatch: got {digest}, expected {expected}.")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        if names != FILES:
            sys.exit(f"Unexpected files in the bundle: {sorted(names)}")
        TARGET.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            (TARGET / name).write_bytes(zf.read(name))

    print(f"Model bundle ({len(data) / 1e6:.1f} MB, sha256 {digest[:12]}...) unpacked to {TARGET}")


if __name__ == "__main__":
    main()
