"""
CLIP ViT-L/14 Deepfake Detection Model
Features:
- Frozen CLIP ViT-L/14 image encoder
- 2-layer MLP or linear projection head
- Optional parameter-efficient fine-tuning (LoRA via PEFT) on visual attention projections
"""

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection


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

        # Load CLIP vision encoder with projection.
        # CLIPVisionModelWithProjection is required: it produces image_embeds
        # (projection_dim=768) which the classifier head is sized for.
        # The fallback to CLIPModel.vision_model was removed because it outputs
        # pooler_output (hidden_size=1024), causing a shape mismatch at the head.
        try:
            self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load CLIPVisionModelWithProjection from '%s'. "
                "Ensure the model weights are cached or the HuggingFace Hub is "
                "reachable. Original error: %s" % (model_name, exc)
            ) from exc

        embed_dim = (
            self.backbone.config.projection_dim
            if hasattr(self.backbone.config, "projection_dim")
            else self.backbone.config.hidden_size
        )

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
            except ImportError as exc:
                raise ImportError(
                    "PEFT (peft) is required when use_lora=True but is not installed. "
                    "Install it with: pip install peft"
                ) from exc

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

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "CLIPDeepfakeDetector":
        """Build a detector from a saved training config."""
        config = config or {}
        model_cfg = config.get("model", config) if isinstance(config, dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}

        return cls(
            model_name=model_cfg.get("name", "openai/clip-vit-large-patch14"),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            head_type=model_cfg.get("head_type", "mlp"),
            hidden_dim=model_cfg.get("hidden_dim", 512),
            dropout=model_cfg.get("dropout", 0.2),
            use_lora=model_cfg.get("use_lora", False),
            lora_r=model_cfg.get("lora_r", 8),
            lora_alpha=model_cfg.get("lora_alpha", 16),
            lora_dropout=model_cfg.get("lora_dropout", 0.1),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device = torch.device("cpu"),
    ) -> Optional["CLIPDeepfakeDetector"]:
        """Load a trained detector, or return None when the checkpoint is absent."""
        path = Path(checkpoint_path)
        if not path.exists():
            return None

        from src.common.utils import load_checkpoint_dict

        checkpoint = load_checkpoint_dict(path, device=device)
        model = cls.from_config(checkpoint.get("config", {}))
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        model.load_state_dict(state_dict)
        return model.to(device).eval()

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
