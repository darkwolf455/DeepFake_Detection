"""
Explainability Module: Grad-CAM Saliency Maps
Generates spatial heatmaps indicating which image or video frame regions
(e.g. blend boundaries, eye reflections, hair artifacts) influenced the deepfake classification.
"""

from typing import Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class ViTGradCAM:
    """
    Computes visual saliency maps for Vision Transformers by extracting
    gradients and activations from the final Transformer layer's attention tokens.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        if self.target_layer is not None:
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0]
            else:
                self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Computes 2D heatmap for input tensor (1, C, H, W).
        Returns normalized 2D numpy array [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor.requires_grad = True
        logits = self.model(input_tensor)
        score = logits[0, target_class]
        score.backward()

        if self.gradients is not None and self.activations is not None:
            # Gradients: (1, N_tokens, Dim), Activations: (1, N_tokens, Dim)
            grads = self.gradients
            acts = self.activations

            # Skip CLS token (token 0)
            if grads.dim() == 3 and grads.size(1) > 1:
                patch_grads = grads[:, 1:, :]
                patch_acts = acts[:, 1:, :]

                # Global average pooling over channels
                weights = torch.mean(patch_grads, dim=1, keepdim=True)
                cam = torch.sum(patch_acts * weights, dim=-1)  # (1, N_patches)
                cam = F.relu(cam)

                # Reshape to 2D grid (e.g. 14x14 for 224x224 patch14)
                grid_size = int(np.sqrt(cam.size(1)))
                cam = cam.view(1, 1, grid_size, grid_size)
                cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
                cam_np = cam.squeeze().detach().cpu().numpy()
            else:
                cam_np = np.ones((224, 224), dtype=np.float32) * 0.5
        else:
            # High-frequency gradient fallback
            grad = input_tensor.grad.squeeze().abs().mean(dim=0).detach().cpu().numpy()
            cam_np = cv2.GaussianBlur(grad, (15, 15), 0)

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


def generate_heatmap_overlay(
    rgb_image: np.ndarray,
    cam_mask: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlays colored heatmap on original RGB image.
    Returns RGB uint8 image.
    """
    h, w, _ = rgb_image.shape
    resized_cam = cv2.resize(cam_mask, (w, h))

    # Convert to 8-bit heatmap
    heatmap_uint8 = np.uint8(255 * resized_cam)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Alpha blend
    blended = np.uint8(alpha * colored_heatmap + (1.0 - alpha) * rgb_image)
    return blended
