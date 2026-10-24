"""
Unit tests for MIME inspection, magic bytes, and file hashing.
"""

import pytest
from fastapi import HTTPException

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.preprocessing import detect_file_type_and_modality


def test_detect_jpeg_magic_bytes():
    jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
    mime, modality = detect_file_type_and_modality(jpeg_header, "photo.jpg")
    assert mime == "image/jpeg"
    assert modality == "image"


def test_detect_png_magic_bytes():
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    mime, modality = detect_file_type_and_modality(png_header, "graphic.png")
    assert mime == "image/png"
    assert modality == "image"


def test_detect_mp4_magic_bytes():
    mp4_header = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00"
    mime, modality = detect_file_type_and_modality(mp4_header, "clip.mp4")
    assert mime == "video/mp4"
    assert modality == "video"


def test_detect_wav_magic_bytes():
    wav_header = b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00"
    mime, modality = detect_file_type_and_modality(wav_header, "speech.wav")
    assert mime == "audio/wav"
    assert modality == "audio"


def test_unsupported_magic_bytes_raises():
    bad_bytes = b"CORRUPTED_GARBAGE_PAYLOAD_1234567890"
    with pytest.raises(HTTPException):
        detect_file_type_and_modality(bad_bytes, "unknown.xyz")
