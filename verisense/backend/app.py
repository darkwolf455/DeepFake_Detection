"""
VeriSense FastAPI Application
Handles multimodal file uploads, model routing, health checks, and heatmap artifacts.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.inference import InferenceManager
from backend.preprocessing import SafeTempFile
from backend.schemas import AnalyzeResponse, HealthResponse
from backend.settings import settings


# Global Inference Manager
inference_manager = InferenceManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load checkpoints and initialize models once
    inference_manager.load_models()
    yield
    # Shutdown logic if needed
    print("[VeriSense] Server shutting down cleanly.")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Multimodal deepfake detection REST API for images, video sequences, and audio.",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://0.0.0.0:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": f"ERR_{exc.status_code}"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Sanitize stack traces to avoid leaking internal system details
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred during forensic analysis.", "error_code": "ERR_INTERNAL_500"},
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and GPU/CPU device telemetry."""
    return HealthResponse(
        status="ok" if inference_manager.active_branches else "degraded",
        device=str(inference_manager.device),
        active_branches=inference_manager.active_branches,
        missing_checkpoints=inference_manager.missing_checkpoints,
        version=settings.version,
    )


@app.get("/api/models")
async def list_models():
    """Lists loaded forensic neural network architectures and checkpoint availability."""
    models = [
        {"name": "CLIP ViT-L/14 Face Classifier", "type": "image",
         "status": "active" if "image" in inference_manager.active_branches else "unavailable"},
        {"name": "CLIP Temporal Classifier" if inference_manager.video_model is not None else "CLIP Sampled-Frame Video Baseline", "type": "video",
         "decision_threshold": inference_manager.video_threshold,
         "status": "active" if "video" in inference_manager.active_branches else "unavailable"},
        {"name": "Wav2Vec2 Base Speech Classifier", "type": "audio",
         "decision_threshold": inference_manager.audio_threshold,
         "status": "active" if "audio" in inference_manager.active_branches else "unavailable"},
        {"name": "LAV-DF Five-Score Fusion", "type": "fusion",
         "decision_threshold": inference_manager.fusion_model.decision_threshold if inference_manager.fusion_model is not None else None,
         "status": "active" if "fusion" in inference_manager.active_branches else "unavailable"},
    ]
    return {"models": models, "device": str(inference_manager.device)}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_media(file: UploadFile = File(...), use_fusion: bool = False):
    """
    Analyzes uploaded media file (Image, Video, or Audio).
    Bounds uploads, routes to an available trained model, and returns an uncalibrated score.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file missing valid filename.")

    # Keep demo uploads bounded to 25 MB for both images and video.
    max_size = settings.max_image_size

    async with SafeTempFile(file, max_bytes=max_size) as (temp_path, mime_type, modality, file_hash):
        response = await inference_manager.analyze_file(temp_path, modality, file_hash, use_fusion=use_fusion)
        return response
