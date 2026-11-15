"""Trained face-image, temporal video and short speech inference."""

import asyncio
import logging
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.schemas import AnalyzeResponse, ModalityScores, ModelVersion, VerdictEnum
from backend.settings import settings
from backend.fusion import load_fusion
from src.image.model import CLIPDeepfakeDetector
from src.audio.model import Wav2Vec2DeepfakeDetector
from src.audio.preprocessing import prepare_audio
from src.video.face_extractor import FaceExtractor, sample_video_frames
from src.video.model import VideoTemporalDetector


class InferenceManager:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
        self.face_extractor = FaceExtractor()
        self.image_model = None
        self.video_model = None
        self.video_threshold = None
        self.audio_model = None
        self.audio_threshold = None
        self.fusion_model = None
        self.fusion_scorer = None
        self.active_branches = []
        self.missing_checkpoints = ["image", "video", "audio", "fusion"]
        self.models_loaded = False

    def load_models(self):
        self.image_model = None
        self.video_model = None
        self.video_threshold = None
        self.audio_model = None
        self.audio_threshold = None
        self.fusion_model = None
        self.fusion_scorer = None
        self.active_branches = []
        self.missing_checkpoints = ["image", "video", "audio", "fusion"]
        try:
            # Read metadata and trained head before downloading the frozen backbone.
            checkpoint = torch.load(settings.image_checkpoint, map_location="cpu", weights_only=True)
            if checkpoint["class_names"] != ["real", "manipulated"] or checkpoint["image_size"] != 224:
                raise ValueError("Unsupported checkpoint labels or image size")
            model = CLIPDeepfakeDetector(
                model_name=checkpoint["model_name"],
                head_type=checkpoint["head_type"],
                hidden_dim=checkpoint["hidden_dim"],
                dropout=checkpoint["dropout"],
            )
            model.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
            self.image_model = model.to(self.device).eval()
            self.active_branches = ["image", "video"]
            self.missing_checkpoints.remove("image")
            self.missing_checkpoints.remove("video")
        except Exception:
            logging.exception("Image detector unavailable; check image_head.pt and the CLIP download")
        if self.image_model is not None and settings.video_checkpoint.exists():
            try:
                checkpoint = torch.load(settings.video_checkpoint, map_location="cpu", weights_only=True)
                config = checkpoint["config"]["temporal_model"]
                if config["input_dim"] != 768:
                    raise ValueError("Video model requires 768-dimensional CLIP features")
                threshold = checkpoint.get("decision_threshold")
                if threshold is not None and (
                    type(threshold) not in (float, int) or not 0 < threshold < 1
                    or checkpoint.get("threshold_source") != "validation"
                ):
                    raise ValueError("Video cutoff must be a finite value between 0 and 1 selected on validation")
                model = VideoTemporalDetector.from_config(config)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                self.video_model = model.to(self.device).eval()
                self.video_threshold = threshold
            except Exception:
                logging.exception("Temporal checkpoint unavailable; retaining the labeled frame-mean baseline")
        if settings.audio_checkpoint.exists():
            try:
                import soundfile
                from scipy.signal import resample_poly
                checkpoint = torch.load(settings.audio_checkpoint, map_location="cpu", weights_only=True)
                model = Wav2Vec2DeepfakeDetector.from_checkpoint(checkpoint)
                self.audio_model = model.to(self.device).eval()
                self.audio_threshold = checkpoint["decision_threshold"]
                self.active_branches.append("audio")
                self.missing_checkpoints.remove("audio")
            except Exception:
                logging.exception("Audio detector unavailable; check audio_head.pt and the audio dependencies")
        if settings.fusion_checkpoint.exists() and all(model is not None for model in
                (self.image_model, self.video_model, self.audio_model)):
            try:
                self.fusion_model, self.fusion_scorer = load_fusion(
                    settings.fusion_checkpoint,
                    {"image_head.pt": settings.image_checkpoint, "video_best.pt": settings.video_checkpoint,
                     "audio_head.pt": settings.audio_checkpoint},
                    self.image_model, self.video_model, self.audio_model, self.device, self.face_extractor)
                self.active_branches.append("fusion")
                self.missing_checkpoints.remove("fusion")
            except Exception:
                logging.exception("Fusion unavailable; check checkpoints/lavdf, decoder and training provenance")
        self.models_loaded = True

    async def analyze_file(self, file_path: Path, modality: str, file_hash: str, use_fusion=False) -> AnalyzeResponse:
        if use_fusion:
            if modality != "video":
                raise HTTPException(400, "Combined video/audio analysis requires a video upload.")
            if getattr(self, "fusion_model", None) is None:
                raise HTTPException(503, "Fusion unavailable. Check checkpoints/lavdf and the server startup log.")
            async with self.semaphore:
                return await asyncio.to_thread(self._analyze_fusion, file_path, file_hash)
        if modality == "audio" and self.audio_model is None:
            raise HTTPException(503, "Audio detector unavailable. Check audio_head.pt and the server startup log.")
        if modality in ("image", "video") and self.image_model is None:
            raise HTTPException(503, "Image detector unavailable. Check the server startup log and image_head.pt.")
        routes = {"image": self._analyze_image, "video": self._analyze_video, "audio": self._analyze_audio}
        if modality not in routes:
            raise HTTPException(400, "Unsupported analysis modality.")
        async with self.semaphore:
            return await asyncio.to_thread(routes[modality], file_path, file_hash)

    def _analyze_image(self, file_path: Path, file_hash: str) -> AnalyzeResponse:
        started = time.perf_counter()
        try:
            with Image.open(file_path) as image:
                if image.width * image.height > 25_000_000:
                    raise HTTPException(413, "Image exceeds the 25-megapixel decoding limit.")
                rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "The uploaded image could not be decoded.") from exc

        faces = self.face_extractor.extract_faces(rgb)
        if len(faces) != 1:
            raise HTTPException(422, f"Expected one detected face, found {len(faces)}. Upload a clear photo of one face.")
        probability = self._score_face(faces[0]["cropped_face"])
        return self._response(probability, "image", file_hash, started)

    def _face_tensor(self, crop):
        # Match the face crop, scaling and normalization used during training.
        pixels = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        return ((pixels - mean) / std).unsqueeze(0).to(self.device)

    def _score_face(self, crop):
        with torch.inference_mode():
            logits = self.image_model(self._face_tensor(crop))
            probability = float(torch.softmax(logits, dim=-1)[0, 1].item())
        if not np.isfinite(probability):
            raise HTTPException(500, "Image detector returned an invalid score.")
        return probability

    def _analyze_video(self, file_path: Path, file_hash: str) -> AnalyzeResponse:
        started = time.perf_counter()
        timeline = []
        crops, timestamps = [], []
        sampled = 0
        temporal = getattr(self, "video_model", None)
        try:
            for rgb, timestamp in sample_video_frames(file_path, max_frames=temporal.max_frames if temporal else 16):
                sampled += 1
                faces = self.face_extractor.extract_faces(rgb)
                if len(faces) == 1:
                    crop = faces[0]["cropped_face"]
                    crops.append(crop)
                    timestamps.append(timestamp)
                    if temporal is None:
                        timeline.append({"t": timestamp, "fake_prob": self._score_face(crop)})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not crops:
            raise HTTPException(422, "No sampled frames contained exactly one clear face.")
        if temporal is not None:
            with torch.inference_mode():
                pixels = torch.cat([self._face_tensor(crop) for crop in crops])
                features = self.image_model.extract_features(pixels).unsqueeze(0)
                mask = torch.ones((1, len(crops)), dtype=torch.bool, device=self.device)
                video_logits, frame_logits = temporal(features, mask=mask)
                probability = float(video_logits.softmax(-1)[0, 1].item())
                scores = frame_logits.softmax(-1)[0, :, 1].cpu().tolist()
            if not np.isfinite([probability, *scores]).all():
                raise HTTPException(500, "Temporal detector returned an invalid score.")
            timeline = [{"t": t, "fake_prob": score} for t, score in zip(timestamps, scores)]
            warnings = [f"Temporal model scored {len(crops)} of {sampled} decoded samples from the first 30 seconds at most.",
                        "Frame scores use video-level training labels; audio and lip-sync are not analyzed."]
            return self._response(probability, "video", file_hash, started, timeline, warnings,
                                  decision_threshold=getattr(self, "video_threshold", None))
        warnings = [
            f"Video baseline: scored {len(timeline)} of {sampled} decoded sampled frames from the first 30 seconds at most.",
            "Video score is the mean of sampled face scores; no temporal, audio or lip-sync model is used.",
        ]
        probability = float(np.mean([point["fake_prob"] for point in timeline]))
        return self._response(probability, "video", file_hash, started, timeline, warnings)

    def _analyze_audio(self, file_path: Path, file_hash: str) -> AnalyzeResponse:
        import soundfile as sf

        started = time.perf_counter()
        try:
            with sf.SoundFile(file_path) as audio:
                rate = audio.samplerate
                if not 8000 <= rate <= 192000 or not 1 <= audio.channels <= 2:
                    raise HTTPException(422, "Use mono or stereo audio sampled between 8 and 192 kHz.")
                if audio.frames > rate * 30:
                    raise HTTPException(413, "Audio demo accepts clips up to 30 seconds.")
                waveform = audio.read(frames=rate * 30 + 1, dtype="float32", always_2d=True)
                if len(waveform) > rate * 30:
                    raise HTTPException(413, "Audio demo accepts clips up to 30 seconds.")
            waveform = prepare_audio(waveform, rate)
        except (RuntimeError, OSError, ValueError) as exc:
            raise HTTPException(400, "Audio could not be decoded. Try a valid WAV or FLAC speech clip.") from exc
        if np.max(np.abs(waveform)) <= 1e-6:
            raise HTTPException(422, "The analyzed audio segment is silent. Upload a clear speech clip.")
        with torch.inference_mode():
            logits = self.audio_model(torch.from_numpy(waveform).unsqueeze(0).to(self.device))
            probability = float(logits.softmax(-1)[0, 1].item())
        if not np.isfinite(probability):
            raise HTTPException(500, "Audio detector returned an invalid score.")
        return self._response(probability, "audio", file_hash, started, warnings=[
            "Audio model analyzes the central 4 seconds; longer recordings are not checked in full.",
        ], decision_threshold=self.audio_threshold)

    def _analyze_fusion(self, file_path: Path, file_hash: str) -> AnalyzeResponse:
        started = time.perf_counter()
        try:
            values = self.fusion_scorer.score(file_path)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, "Combined analysis requires at least four clear single-face samples and a "
                                "decodable, non-silent soundtrack up to 30 seconds. "
                                "Try a clearer clip or turn off combined analysis for visual-only scoring.") from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(504, "Video soundtrack decoding timed out.") from exc
        from src.fusion.model import FEATURE_COLUMNS
        with torch.inference_mode():
            probability = float(self.fusion_model([values[name] for name in FEATURE_COLUMNS]).item())
        scores = ModalityScores(image=round(values["image_max"], 4), video=round(values["video_score"], 4),
                                audio=round(values["audio_max"], 4), fusion=round(probability, 4))
        models = [ModelVersion(name=name, version="lavdf-run2" if kind != "video" else "student-v1", modality=kind)
                  for name, kind in [("CLIP Face Classifier (LAV-DF)", "image"),
                    ("CLIP Temporal Classifier", "video"), ("Wav2Vec2 Speech Classifier (LAV-DF)", "audio"),
                    ("LAV-DF Five-Score Fusion", "fusion")]]
        return self._response(probability, "video", file_hash, started, scores=scores, models=models,
            decision_threshold=self.fusion_model.decision_threshold, warnings=[
                f"Combined analysis scored {values['face_frames']} face frames and {values['scored_audio_windows']} "
                "four-second speech windows; uploads must be at most 30 seconds.",
                "Image/audio cards show maximum sampled scores. Fusion also uses their means and the temporal score.",
                "Lip-sync is not analyzed. No fusion verdict is produced when a required modality cannot be scored."])

    def _response(self, probability, modality, file_hash, started, timeline=None, warnings=None, decision_threshold=None,
                  scores=None, models=None):
        fused = scores is not None and scores.fusion is not None
        if modality in ("video", "audio") and decision_threshold is not None:
            verdict = VerdictEnum.MANIPULATED if probability >= decision_threshold else VerdictEnum.AUTHENTIC
            confidence = probability if verdict == VerdictEnum.MANIPULATED else 1 - probability
            warnings = (warnings or []) + [
                f"{'Fusion' if fused else modality.title()} uses a validation-selected cutoff of {decision_threshold:.6f}; scores are uncalibrated."
            ]
        elif settings.inconclusive_lower <= probability <= settings.inconclusive_upper:
            verdict = VerdictEnum.INCONCLUSIVE
            confidence = 0.0  # No confident decision within the uncalibrated margin.
        else:
            verdict = VerdictEnum.MANIPULATED if probability > settings.inconclusive_upper else VerdictEnum.AUTHENTIC
            confidence = max(probability, 1 - probability)

        return AnalyzeResponse(
            id=uuid.uuid4().hex,
            modality=modality,
            verdict=verdict,
            confidence=round(confidence, 4),
            scores=scores if scores is not None else ModalityScores(**{modality: round(probability, 4)}),
            timeline=timeline,
            models=models if models is not None else [ModelVersion(
                name="Wav2Vec2 Base Speech Classifier" if modality == "audio" else "CLIP Face Classifier" if modality == "image" else (
                    "CLIP + Temporal Sequence Classifier" if getattr(self, "video_model", None) is not None else "CLIP Face Classifier + Frame Mean"),
                version="student-v1", modality=modality)],
            processing_ms=int((time.perf_counter() - started) * 1000),
            file_sha256=file_hash,
            warnings=(warnings or []) + [
                ("Experimental fusion evaluated on a small LAV-DF subset; video-only manipulations were frequently missed."
                 if fused else "Student model trained on a small ASVspoof 2019 LA subset; unfamiliar synthetic voices may be missed."
                 if modality == "audio" else
                 "Student model trained on a small FaceForensics++ Deepfakes subset; unfamiliar edits may be missed."),
                "Scores and decision margins are not calibrated probabilities or proof of authenticity.",
            ],
        )
