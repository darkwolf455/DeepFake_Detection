"""API contract tests without downloads; model outputs are controlled test fixtures."""
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import app, inference_manager


@pytest.fixture
def client(monkeypatch):
    def no_download():
        inference_manager.image_model = None
        inference_manager.video_model = None
        inference_manager.video_threshold = None
        inference_manager.audio_model = None
        inference_manager.audio_threshold = None
        inference_manager.fusion_model = None
        inference_manager.fusion_scorer = None
        inference_manager.active_branches = []
        inference_manager.missing_checkpoints = ["image", "video", "audio", "fusion"]
    monkeypatch.setattr(inference_manager, "load_models", no_download)
    with TestClient(app) as test_client:
        yield test_client


def jpeg():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), color=(73, 109, 137)).save(buffer, format="JPEG")
    return buffer.getvalue()


def ready(monkeypatch, logits=(0.0, 2.0), faces=1):
    monkeypatch.setattr(inference_manager, "image_model", lambda pixels: torch.tensor([logits]))
    monkeypatch.setattr(inference_manager.face_extractor, "extract_faces", lambda rgb: [
        {"cropped_face": np.zeros((224, 224, 3), dtype=np.uint8)} for _ in range(faces)
    ])


def test_missing_model_is_reported_honestly(client):
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["active_branches"] == []
    assert all(model["status"] == "unavailable" for model in client.get("/api/models").json()["models"])
    response = client.post("/api/analyze", files={"file": ("face.jpg", jpeg(), "image/jpeg")})
    assert response.status_code == 503
    assert response.json()["error_code"] == "ERR_503"


@pytest.mark.parametrize("modality, data, filename", [
    ("audio", b"RIFF1234WAVEfmt ", "speech.wav"),
    ("video", b"\x00\x00\x00\x20ftypisom", "clip.mp4"),
])
def test_unfinished_modalities_return_unavailable(client, modality, data, filename):
    response = client.post("/api/analyze", files={"file": (filename, data)})
    assert response.status_code == 503


@pytest.mark.parametrize("faces", [0, 2])
def test_requires_one_face(client, monkeypatch, faces):
    ready(monkeypatch, faces=faces)
    response = client.post("/api/analyze", files={"file": ("face.jpg", jpeg())})
    assert response.status_code == 422
    assert f"found {faces}" in response.json()["detail"]


@pytest.mark.parametrize("logits, verdict", [
    ((0.0, 2.0), "manipulated"), ((2.0, 0.0), "authentic"), ((0.0, 0.0), "inconclusive"),
])
def test_returns_only_the_image_model_score(client, monkeypatch, logits, verdict):
    ready(monkeypatch, logits=logits)
    content = jpeg()
    response = client.post("/api/analyze", files={"file": ("face.jpg", content)})
    assert response.status_code == 200
    result = response.json()
    assert result["verdict"] == verdict
    assert result["scores"]["image"] == round(torch.softmax(torch.tensor(logits), dim=-1)[1].item(), 4)
    assert result["scores"]["audio"] is None
    assert result["heatmap_url"] is None
    assert result["file_sha256"] == hashlib.sha256(content).hexdigest()
    assert result["warnings"]
    if verdict == "inconclusive":
        assert result["confidence"] == 0


def test_invalid_image_returns_readable_error(client, monkeypatch):
    ready(monkeypatch)
    response = client.post("/api/analyze", files={"file": ("bad.png", b"\x89PNG\r\n\x1a\ninvalid")})
    assert response.status_code == 400


def test_invalid_model_score_is_rejected(client, monkeypatch):
    ready(monkeypatch, logits=(float("nan"), 1.0))
    response = client.post("/api/analyze", files={"file": ("face.jpg", jpeg())})
    assert response.status_code == 500
    assert "invalid score" in response.json()["detail"]


def test_requested_fusion_is_unavailable_without_its_checkpoint(client):
    response = client.post("/api/analyze?use_fusion=true", files={"file": ("clip.mp4", b"\x00\x00\x00\x20ftypisom")})
    assert response.status_code == 503
    assert "Fusion unavailable" in response.json()["detail"]


def test_fusion_query_routes_video_and_reports_its_score_and_status(client, monkeypatch):
    from types import SimpleNamespace
    from src.fusion.model import FEATURE_COLUMNS, LateFusionClassifier
    from test_fusion_inference import classifier_checkpoint
    model = LateFusionClassifier(classifier_checkpoint())
    values = dict.fromkeys(FEATURE_COLUMNS, .7) | dict(face_frames=8, scored_audio_windows=3)
    monkeypatch.setattr(inference_manager, "fusion_model", model)
    monkeypatch.setattr(inference_manager, "fusion_scorer", SimpleNamespace(score=lambda path: values))
    monkeypatch.setattr(inference_manager, "active_branches", ["fusion"])
    response = client.post("/api/analyze?use_fusion=true", files={"file": ("clip.mp4", b"\x00\x00\x00\x20ftypisom")})
    assert response.status_code == 200
    result = response.json()
    assert result["scores"]["fusion"] == round(float(model([.7] * 5)), 4)
    assert result["scores"]["av_sync"] is None
    assert result["modality"] == "video"
    fusion = next(item for item in client.get("/api/models").json()["models"] if item["type"] == "fusion")
    assert fusion["status"] == "active" and fusion["decision_threshold"] == .5
    invalid = client.post("/api/analyze?use_fusion=true", files={"file": ("face.jpg", jpeg())})
    assert invalid.status_code == 400
