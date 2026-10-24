"""
Pydantic Request & Response Data Contracts for VeriSense API
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ModalityEnum(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class VerdictEnum(str, Enum):
    AUTHENTIC = "authentic"
    MANIPULATED = "manipulated"
    INCONCLUSIVE = "inconclusive"


class TimelinePoint(BaseModel):
    t: float = Field(..., description="Timestamp in seconds")
    fake_prob: float = Field(..., ge=0.0, le=1.0, description="Manipulated probability at timestamp")


class ModelVersion(BaseModel):
    name: str
    version: str
    modality: str
    status: str = "active"


class ModalityScores(BaseModel):
    image: Optional[float] = Field(None, ge=0.0, le=1.0)
    video: Optional[float] = Field(None, ge=0.0, le=1.0)
    audio: Optional[float] = Field(None, ge=0.0, le=1.0)
    av_sync: Optional[float] = Field(None, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    id: str = Field(..., description="Unique analysis identifier")
    modality: ModalityEnum
    verdict: VerdictEnum
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated decision confidence")
    scores: ModalityScores
    timeline: Optional[List[TimelinePoint]] = None
    heatmap_url: Optional[str] = None
    models: List[ModelVersion]
    processing_ms: int
    warnings: Optional[List[str]] = None
    file_sha256: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    active_branches: List[str]
    missing_checkpoints: List[str]
    version: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
