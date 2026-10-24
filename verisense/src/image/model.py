"""
CLIP ViT-L/14 Deepfake Detection Model
Features:
- Frozen CLIP ViT-L/14 image encoder
- 2-layer MLP or linear projection head
- Optional parameter-efficient fine-tuning (LoRA via PEFT) on visual attention projections
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPVisionModelWithProjection


class CLIPDeepfakeDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        freeze_backbone: bool = True,
        head_type: str = "mlp",
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora

        # Load CLIP vision encoder with projection
        try:
            self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name)
        except Exception:
            # Fallback to general CLIPModel vision component
            full_model = CLIPModel.from_pretrained(model_name)
            self.backbone = full_model.vision_model

        embed_dim = self.backbone.config.projection_dim if hasattr(self.backbone.config, "projection_dim") else self.backbone.config.hidden_size

        if freeze_backbone and not use_lora:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, peft_config)
                self.backbone.print_trainable_parameters()
            except ImportError:
                print("Warning: PEFT library not installed. Continuing with frozen backbone.")
                for param in self.backbone.parameters():
                    param.requires_grad = False

        # Classification Head
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(vision_outputs, "image_embeds"):
            features = vision_outputs.image_embeds
        elif hasattr(vision_outputs, "pooler_output"):
            features = vision_outputs.pooler_output
        else:
            features = vision_outputs[0][:, 0, :]  # CLS token

        logits = self.head(features)
        return logits

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(vision_outputs, "image_embeds"):
            features = vision_outputs.image_embeds
        elif hasattr(vision_outputs, "pooler_output"):
            features = vision_outputs.pooler_output
        else:
            features = vision_outputs[0][:, 0, :]
        return features / features.norm(dim=-1, keepdim=True)
