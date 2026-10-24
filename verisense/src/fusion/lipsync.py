"""
Audio-Visual Lip-Sync Consistency Analysis (SyncNet-style architecture)
Computes cross-modal embedding similarity between visual mouth motion and speech acoustics.
Mismatches or temporal offset indicate deepfake voice dubbing or lip manipulation (e.g. Wav2Lip).
"""

from typing import List, Optional
import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LipSyncScorer(nn.Module):
    """
    Evaluates synchronization consistency between video mouth crops and audio speech.
    Uses dual-tower encoders:
    1. Visual Tower: 3D/2D CNN over 5 consecutive mouth ROI frames
    2. Audio Tower: 2D CNN over Mel-spectrogram chunk corresponding to the 5 frames (0.2s)
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Visual sub-network (input: 5 frames, 112x112 mouth ROI)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=5, stride=2, padding=2),  # 5 frames * 3 channels = 15
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

        # Audio sub-network (input: 1 channel Mel-spectrogram, 80 mel bins x 20 time frames)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, mouth_frames: torch.Tensor, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between visual mouth and acoustic representations.
        Returns:
            similarity: (Batch,) scalar between -1 and 1
        """
        v_feat = self.visual_encoder(mouth_frames)
        v_feat = F.normalize(v_feat, p=2, dim=-1)

        a_feat = self.audio_encoder(mel_spectrogram)
        a_feat = F.normalize(a_feat, p=2, dim=-1)

        similarity = torch.sum(v_feat * a_feat, dim=-1)
        return similarity

    @torch.no_grad()
    def compute_sync_score(
        self,
        face_crops: List[np.ndarray],
        waveform: np.ndarray,
        sr: int = 16000,
        fps: float = 25.0,
    ) -> float:
        """
        Computes average lip-sync anomaly score (0 = perfectly synchronized, 1 = severely desynchronized/deepfake).
        """
        if len(face_crops) < 5 or len(waveform) < sr * 0.2:
            return 0.5  # Neutral default when video too short

        # Extract lower-third mouth regions from face crops
        mouth_crops = []
        for face in face_crops[:15]:
            h, w, _ = face.shape
            # Lower 40% of the face contains the mouth
            mouth = face[int(h * 0.6) : h, int(w * 0.2) : int(w * 0.8)]
            if mouth.size > 0:
                mouth = cv2.resize(mouth, (112, 112))
                mouth_crops.append(mouth)

        if len(mouth_crops) < 5:
            return 0.5

        # Stack 5 consecutive frames
        five_mouths = np.concatenate(mouth_crops[:5], axis=-1)  # (112, 112, 15)
        mouth_tensor = torch.from_numpy(five_mouths).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # Extract Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=waveform[: int(sr * 0.2)], sr=sr, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = cv2.resize(mel_db, (20, 80))
        audio_tensor = torch.from_numpy(mel_resized).float().unsqueeze(0).unsqueeze(0)

        # Measure similarity
        cos_sim = self(mouth_tensor, audio_tensor).item()

        # Invert cosine similarity: high sync similarity -> low fake score
        # Cosine similarity typically spans 0.0 to 0.7 for authentic speech
        fake_score = np.clip(1.0 - (cos_sim + 0.3) / 1.0, 0.0, 1.0)
        return float(fake_score)
