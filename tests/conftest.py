import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "model_meta.json"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="train a model first: python -m src.models.train")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import create_app
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture(scope="session")
def model():
    import json
    import joblib
    key = json.loads(MODEL.read_text())["model_key"]
    return joblib.load(ROOT / "models" / f"{key}.joblib")
