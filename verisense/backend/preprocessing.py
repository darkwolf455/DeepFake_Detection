"""
Preprocessing and Media Validation:
- Magic-byte file signature and MIME type inspection
- Size limit validation
- Safe temporary file context management with guaranteed unlink
- SHA-256 fingerprinting
"""

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Generator, Tuple
from fastapi import HTTPException, UploadFile


# Magic byte signatures
SIGNATURES = {
    # Images
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # WebP or WAV (checked further)
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    # Audio
    b"ID3": "audio/mpeg",
    b"\xff\xfb": "audio/mpeg",
    b"\xff\xf3": "audio/mpeg",
    b"fLaC": "audio/flac",
    b"OggS": "audio/ogg",
    # Video
    b"ftyp": "video/mp4",
    b"\x1a\x45\xdf\xa3": "video/webm",  # WebM or MKV
}


def detect_file_type_and_modality(head_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Validates file headers against authentic magic bytes.
    Returns:
        (detected_mime, modality: 'image' | 'video' | 'audio')
    """
    ext = Path(filename).suffix.lower()

    # Check MP4/MOV ftyp box in first 32 bytes
    if b"ftyp" in head_bytes[:32]:
        return "video/mp4", "video"

    # Check WebM / MKV
    if head_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        if ext in [".webm", ".mkv"]:
            return "video/webm", "video"

    # Check RIFF header (can be WEBP or WAV)
    if head_bytes.startswith(b"RIFF"):
        if b"WEBP" in head_bytes[:16]:
            return "image/webp", "image"
        if b"WAVE" in head_bytes[:16]:
            return "audio/wav", "audio"

    # Check JPEG
    if head_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "image"

    # Check PNG
    if head_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "image"

    # Check Audio signatures
    if head_bytes.startswith(b"ID3") or head_bytes.startswith(b"\xff\xfb") or head_bytes.startswith(b"\xff\xf3"):
        return "audio/mpeg", "audio"
    if head_bytes.startswith(b"fLaC"):
        return "audio/flac", "audio"
    if head_bytes.startswith(b"OggS"):
        return "audio/ogg", "audio"

    # Fallback checking common extension if signature is generic
    if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        return f"image/{ext.replace('.', '')}", "image"
    if ext in [".mp4", ".mov", ".avi", ".webm", ".mkv"]:
        return "video/mp4", "video"
    if ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
        return "audio/wav", "audio"

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported or unrecognizable file format for {filename}. Magic bytes do not match valid media.",
    )


class SafeTempFile:
    """Context manager for temporary file creation with guaranteed cleanup."""

    def __init__(self, upload_file: UploadFile, max_bytes: int):
        self.upload_file = upload_file
        self.max_bytes = max_bytes
        self.temp_path: Path | None = None
        self.sha256 = hashlib.sha256()

    async def __aenter__(self) -> Tuple[Path, str, str, str]:
        suffix = Path(self.upload_file.filename).suffix
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self.temp_path = Path(temp.name)

        bytes_written = 0
        head_bytes = b""

        while chunk := await self.upload_file.read(1024 * 64):
            bytes_written += len(chunk)
            if bytes_written > self.max_bytes:
                temp.close()
                self.temp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum permissible size of {self.max_bytes // (1024*1024)}MB.",
                )
            if len(head_bytes) < 64:
                head_bytes += chunk[: 64 - len(head_bytes)]

            self.sha256.update(chunk)
            temp.write(chunk)

        temp.close()

        mime_type, modality = detect_file_type_and_modality(head_bytes, self.upload_file.filename)
        file_hash = self.sha256.hexdigest()

        return self.temp_path, mime_type, modality, file_hash

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.temp_path and self.temp_path.exists():
            try:
                self.temp_path.unlink()
            except Exception:
                pass
