"""Five-score regularized late fusion. Scores are not probability-calibrated."""
import torch
from torch import nn

FEATURE_COLUMNS = ("image_mean", "image_max", "video_score", "audio_mean", "audio_max")


class LateFusionClassifier(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        if (checkpoint["model_type"] != "standardized_logistic_regression"
                or checkpoint["feature_columns"] != list(FEATURE_COLUMNS)
                or checkpoint["class_names"] != ["real", "manipulated"]
                or checkpoint["threshold_source"] != "validation"):
            raise ValueError("Unsupported fusion checkpoint or feature order")
        self.decision_threshold = float(checkpoint["decision_threshold"])
        if not 0 < self.decision_threshold < 1:
            raise ValueError("Invalid fusion cutoff")
        for name in ("mean", "scale", "weight", "bias"):
            value = torch.as_tensor(checkpoint[name], dtype=torch.float64).detach().clone()
            if value.shape != ((1,) if name == "bias" else (len(FEATURE_COLUMNS),)) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid fusion parameter: {name}")
            self.register_buffer(name, value)
        if not (self.scale > 0).all():
            raise ValueError("Fusion feature scales must be positive")

    def forward(self, scores):
        scores = (scores.to(self.mean) if torch.is_tensor(scores) else
                  torch.tensor(scores, dtype=self.mean.dtype, device=self.mean.device))
        if (scores.ndim not in (1, 2) or scores.shape[-1] != len(FEATURE_COLUMNS)
                or not torch.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all()):
            raise ValueError("Fusion requires five finite branch scores between zero and one")
        return torch.sigmoid(((scores - self.mean) / self.scale) @ self.weight + self.bias[0])
