"""
VeriSense FastAPI Application
Handles multimodal file uploads, model routing, health checks, and heatmap artifacts.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.inference import InferenceManager
from backend.preprocessing import SafeTempFile
from backend.schemas import AnalyzeResponse, ErrorResponse, HealthResponse, ModelVersion
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
    allow_origins=["*"],
    allow_credentials=True,
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
        status="ok",
        device=str(inference_manager.device),
        active_branches=inference_manager.active_branches,
        missing_checkpoints=inference_manager.missing_checkpoints,
        version=settings.version,
    )


@app.get("/api/models")
async def list_models():
    """Lists loaded forensic neural network architectures and checkpoint availability."""
    models = [
        {"name": "CLIP ViT-L/14 Vision Head", "type": "image", "status": "active" if "image" in inference_manager.active_branches else "fallback_active", "checkpoint": str(settings.image_checkpoint)},
        {"name": "Temporal Transformer Sequence Encoder", "type": "video", "status": "active" if "video" in inference_manager.active_branches else "fallback_active", "checkpoint": str(settings.video_checkpoint)},
        {"name": "Wav2Vec2 XLS-R 300M Forensics", "type": "audio", "status": "active" if "audio" in inference_manager.active_branches else "fallback_active", "checkpoint": str(settings.audio_checkpoint)},
        {"name": "Platt-Calibrated Late Fusion MLP", "type": "fusion", "status": "active" if "fusion" in inference_manager.active_branches else "fallback_active", "checkpoint": str(settings.fusion_checkpoint)},
        {"name": "SyncNet Cross-Modal LipSync Scorer", "type": "multimodal", "status": "active", "checkpoint": "internal"},
    ]
    return {"models": models, "device": str(inference_manager.device)}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_media(file: UploadFile = File(...)):
    """
    Analyzes uploaded media file (Image, Video, or Audio).
    Validates magic bytes, extracts forensic artifacts, and computes calibrated deepfake probability.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file missing valid filename.")

    # Maximum file limit: 150MB
    max_size = settings.max_video_size

    async with SafeTempFile(file, max_bytes=max_size) as (temp_path, mime_type, modality, file_hash):
        response = await inference_manager.analyze_file(temp_path, modality, file_hash)
        return response


@app.get("/heatmaps/{heatmap_id}.png")
async def get_heatmap(heatmap_id: str):
    """Retrieves generated Grad-CAM heatmap visualization overlay."""
    file_path = settings.heatmaps_dir / f"{heatmap_id}.png"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Heatmap artifact expired or not found.")
    return FileResponse(path=file_path, media_type="image/png")
