"""
Unit tests for FastAPI endpoints: /health, /api/models, /api/analyze with synthetic inputs.
"""

import io
from fastapi.testclient import TestClient
from PIL import Image
import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "version" in data


def test_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 4


def test_analyze_synthetic_image(client):
    # Generate in-memory valid JPEG
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/analyze",
        files={"file": ("test_synthetic.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["modality"] == "image"
    assert res_data["verdict"] in ["authentic", "manipulated", "inconclusive"]
    assert 0.0 <= res_data["confidence"] <= 1.0
    assert "image" in res_data["scores"]
    assert "file_sha256" in res_data
