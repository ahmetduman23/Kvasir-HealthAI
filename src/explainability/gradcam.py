# src/explainability/gradcam.py
import torch
import numpy as np
import cv2
from typing import Optional, Tuple

class SegGradCAM:
    """
    Grad-CAM for segmentation models (e.g., U-Net).

    Usage:
        target = model.dec4.conv[3]  # or pick the last Conv2d layer
        engine = SegGradCAM(model, target_layer=target, device=DEVICE)
        with torch.enable_grad():
            probs, cam = engine.generate(x)   # x: (N,C,H,W), requires_grad=True
        overlay_rgb = engine.overlay(bgr_uint8, cam, alpha=0.35)
        engine.close()
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None

        # Register hooks (do NOT detach inside hooks)
        self._h_fwd = target_layer.register_forward_hook(self._forward_hook)
        self._h_bwd = target_layer.register_full_backward_hook(self._backward_hook)

    # ---- hooks ----
    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    # ---- lifecycle ----
    def close(self) -> None:
        self._h_fwd.remove()
        self._h_bwd.remove()
        self.activations = None
        self.gradients = None

    # ---- core ----
    def generate(self, x: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Compute probabilities and Grad-CAM for the first sample in the batch.

        Args:
            x: Input tensor (N,C,H,W) already on any device; will be moved to self.device.

        Returns:
            probs (torch.Tensor): sigmoid(logits), shape (N,1,H,W), detached
            cam (np.ndarray): normalized CAM for sample 0, shape (H,W), float32 in [0,1]
        """
        x = x.to(self.device)
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)               # (N,1,H,W)
        probs  = torch.sigmoid(logits)

        # Scalar target for backprop (mean over batch & spatial)
        target = probs.mean()
        target.backward()

        act  = self.activations   # (N,C,H,W)
        grad = self.gradients     # (N,C,H,W)
        assert act is not None and grad is not None, "Grad-CAM hooks failed to capture tensors."

        # Global Average Pooling on gradients → channel weights
        w = grad.mean(dim=(2, 3), keepdim=True)   # (N,C,1,1)
        cam = (w * act).sum(dim=1)                # (N,H,W)
        cam = torch.relu(cam)[0]                  # first sample

        # Normalize to [0,1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return probs.detach(), cam.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def overlay(bgr_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """
        Overlay a CAM heatmap on a BGR image and return RGB for display.
        """
        h, w = bgr_uint8.shape[:2]
        cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heat  = (255 * cam_r).astype(np.uint8)
        heat  = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        out   = cv2.addWeighted(bgr_uint8, 1.0, heat, alpha, 0.0)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
