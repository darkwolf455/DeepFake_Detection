"""
Audio Deepfake Detection Model
Backbone: frozen Wav2Vec2 base with layer-wise learnable weights
Head: Multi-head attentive temporal pooling and MLP classification head
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


class AttentivePooling(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = torch.softmax(self.proj(x), dim=1)  # (B, T, 1)
        pooled = torch.sum(x * weights, dim=1)        # (B, D)
        return pooled


class Wav2Vec2DeepfakeDetector(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
        unfreeze_last_n_layers: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # Load pretrained acoustic model
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.freeze_backbone = unfreeze_last_n_layers == 0
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)

        if freeze_feature_extractor and hasattr(self.backbone, "feature_extractor"):
            for param in self.backbone.feature_extractor.parameters():
                param.requires_grad = False

        # Freeze early transformer encoder layers if desired
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
            total_layers = len(self.backbone.encoder.layers)
            for i in range(total_layers - unfreeze_last_n_layers):
                for p in self.backbone.encoder.layers[i].parameters():
                    p.requires_grad = False

        num_hidden_layers = self.backbone.config.num_hidden_layers + 1
        embed_dim = self.backbone.config.hidden_size

        # Learnable softmax layer weights for combining multi-level acoustic representations
        self.layer_weights = nn.Parameter(torch.ones(num_hidden_layers) / num_hidden_layers)

        self.pooler = AttentivePooling(embed_dim, hidden_dim=hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if (checkpoint["model_name"] != "facebook/wav2vec2-base"
                or checkpoint["class_names"] != ["real", "manipulated"]
                or checkpoint["sample_rate"] != 16000 or checkpoint["duration"] != 4.0):
            raise ValueError("Unsupported audio backbone, labels or preprocessing")
        threshold = checkpoint["decision_threshold"]
        if (type(threshold) not in (float, int) or not 0 < threshold < 1
                or checkpoint.get("threshold_source") != "validation"):
            raise ValueError("Audio cutoff must be between 0 and 1 and selected on validation")
        model = cls(backbone_name=checkpoint["model_name"], hidden_dim=checkpoint["hidden_dim"],
                    dropout=checkpoint["dropout"], unfreeze_last_n_layers=0)
        expected = {key for key in model.state_dict() if not key.startswith("backbone.")}
        if set(checkpoint["head_state_dict"]) != expected:
            raise ValueError("Audio checkpoint must contain exactly the trained pooling and classifier weights")
        model.load_state_dict(checkpoint["head_state_dict"], strict=False)
        return model

    def extract_features(self, input_values: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
        # Request all hidden states from transformer
        outputs = self.backbone(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (B, T, D)

        norm_weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # (Layers, B, T, D)
        weighted_features = torch.sum(stacked * norm_weights.view(-1, 1, 1, 1), dim=0)  # (B, T, D)

        return self.pooler(weighted_features)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(input_values))

    @torch.no_grad()
    def compute_timeline(
        self,
        full_waveform: np.ndarray,
        sr: int = 16000,
        window_sec: float = 4.0,
        hop_sec: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> List[Dict[str, float]]:
        """
        Sliding-window evaluation returning a per-second fake probability timeline.
        """
        self.eval()
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)
        timeline = []

        total_samples = len(full_waveform)
        if total_samples < window_samples:
            # Pad
            padded = np.pad(full_waveform, (0, window_samples - total_samples), mode="wrap")
            tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self(tensor)
            prob = float(torch.softmax(logits, dim=-1)[0, 1].cpu().item())
            return [{"t": 0.0, "fake_prob": round(prob, 4)}]

        for start_idx in range(0, total_samples - window_samples + 1, hop_samples):
            chunk = full_waveform[start_idx : start_idx + window_samples]
            tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self(tensor)
            prob = float(torch.softmax(logits, dim=-1)[0, 1].cpu().item())
            time_sec = round(start_idx / sr, 2)
            timeline.append({"t": time_sec, "fake_prob": round(prob, 4)})

        return timeline
