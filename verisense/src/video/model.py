"""
Video Temporal Detector
Architecture:
- Linear input projection to hidden dimension
- Sinusoidal or learnable positional encoding
- Multi-layer Transformer Encoder (or BiLSTM)
- Attentive temporal pooling head for video-level verdict
- Frame-level classification head for per-frame timeline risk scores
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        scores = self.attn(x)  # (B, T, 1)
        if mask is not None:
            # Mask out padded positions
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        pooled = torch.sum(x * weights, dim=1)  # (B, D)
        return pooled


class VideoTemporalDetector(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        model_type: str = "transformer",
        max_frames: int = 64,
    ):
        super().__init__()
        self.model_type = model_type
        self.input_proj = nn.Linear(input_dim, d_model)

        if model_type == "transformer":
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_frames + 16)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # BiLSTM sequence model
            self.sequence_encoder = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.pooler = AttentionPooling(d_model)
        # Video-level classification head
        self.video_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        # Per-frame timeline classification head
        self.frame_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (Batch, Time, Input_Dim)
            mask: (Batch, Time) boolean mask (True for valid frames, False for padded)
        Returns:
            video_logits: (Batch, 2)
            frame_logits: (Batch, Time, 2)
        """
        x = self.input_proj(embeddings)

        if self.model_type == "transformer":
            x = self.pos_encoder(x)
            # PyTorch TransformerEncoder src_key_padding_mask requires True for padded values
            padding_mask = ~mask if mask is not None else None
            encoded_seq = self.sequence_encoder(x, src_key_padding_mask=padding_mask)
        else:
            encoded_seq, _ = self.sequence_encoder(x)

        # Video-level pooled representation
        video_feat = self.pooler(encoded_seq, mask=mask)
        video_logits = self.video_classifier(video_feat)

        # Frame-level predictions for timeline
        frame_logits = self.frame_classifier(encoded_seq)

        return video_logits, frame_logits
