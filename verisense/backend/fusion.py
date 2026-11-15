"""Load the evaluated LAV-DF heads while sharing the existing frozen backbones."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch
from torch import nn

from src.fusion.build_features import MultimodalScorer, sha256
from src.fusion.model import LateFusionClassifier


class SpeechHeadAdapter(nn.Module):
    def __init__(self, source, head):
        super().__init__()
        self.source, self.head = source, head

    def forward(self, waveform):
        return self.head(self.source.extract_features(waveform))


def load_fusion(checkpoint_path, source_paths, image, video, audio, device, extractor):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classifier = LateFusionClassifier(checkpoint).to(device).eval()
    lineage = checkpoint["provenance"]
    folder = checkpoint_path.parent
    paths = {"image_head.pt": folder / "image_lavdf_head.pt",
             "audio_head.pt": folder / "audio_lavdf_head.pt", "video_best.pt": folder / "video_best.pt"}
    if lineage.get("training_predictions") != "grouped_out_of_fold":
        raise ValueError("Expected the grouped OOF fusion checkpoint")
    if (any(sha256(path) != lineage["checkpoints"][name] for name, path in paths.items())
            or sha256(source_paths["video_best.pt"]) != lineage["checkpoints"]["video_best.pt"]
            or any(sha256(source_paths[name]) != digest for name, digest in lineage["source_checkpoints"].items())):
        raise ValueError("Fusion heads or original models differ from the evaluated weights")
    root = Path(__file__).resolve().parents[1]
    visual_cache, audio_cache = lineage["caches"]["visual"], lineage["caches"]["audio"]
    backend = "mediapipe" if extractor.use_mediapipe else "haar"
    if (visual_cache["face_backend"] != backend
            or visual_cache["face_code_sha256"] != sha256(root / "src/video/face_extractor.py")
            or audio_cache["audio_code_sha256"] != sha256(root / "src/audio/model.py")
            or audio_cache["preprocessing_sha256"] != sha256(root / "src/audio/preprocessing.py")):
        raise ValueError("Fusion face detector or speech preprocessing differs from training")
    visual = torch.load(paths["image_head.pt"], map_location="cpu", weights_only=True)
    speech = torch.load(paths["audio_head.pt"], map_location="cpu", weights_only=True)
    if (visual["model_name"] != image.model_name or visual["image_size"] != 224
            or visual["class_names"] != ["real", "manipulated"]
            or speech["model_name"] != audio.backbone_name or speech["sample_rate"] != 16000
            or speech["duration"] != 4. or speech["class_names"] != ["real", "manipulated"]
            or video.max_frames != 16):
        raise ValueError("Unsupported fusion backbone, labels or sampling")
    frozen = {key: value for key, value in audio.state_dict().items()
              if not key.startswith(("head.", "backbone."))}
    exported = {key: value for key, value in speech["head_state_dict"].items() if not key.startswith("head.")}
    if set(frozen) != set(exported) or any(not torch.equal(value.cpu(), exported[key]) for key, value in frozen.items()):
        raise ValueError("Audio pooling changed; frozen feature reuse is invalid")
    image_head, audio_head = deepcopy(image.head), deepcopy(audio.head)
    image_head.load_state_dict(visual["head_state_dict"], strict=True)
    audio_head.load_state_dict({key.removeprefix("head."): value for key, value in speech["head_state_dict"].items()
                               if key.startswith("head.")}, strict=True)
    image_head = image_head.to(device).eval().requires_grad_(False)
    audio_head = audio_head.to(device).eval().requires_grad_(False)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    scorer = MultimodalScorer(SimpleNamespace(backbone=image.backbone, head=image_head), video,
        SpeechHeadAdapter(audio, audio_head).eval(), device, extractor, ffmpeg)
    return classifier, scorer
