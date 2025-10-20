import numpy as np
import cv2
from typing import Tuple

def overlay_mask(
    img: np.ndarray,
    mask_u8: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
    assume_bgr: bool = False,
) -> np.ndarray:
    """
    Overlay a binary mask (0/255 or 0/1) on an image.

    Args:
        img: HxWx3 RGB (default) or BGR (if assume_bgr=True) uint8 image.
        mask_u8: HxW uint8/bool mask (non-zero treated as foreground).
        alpha: overlay transparency.
        color: overlay color in the same channel order as 'img'.
        assume_bgr: set True if 'img' is BGR (OpenCV read).
    Returns:
        Image with colored overlay in the same color space as input.
    """
    if img.dtype != np.uint8:
        raise ValueError("overlay_mask expects a uint8 image.")

    m = (mask_u8 > 0).astype(np.uint8)
    out = img.copy()
    c = np.array(color, dtype=np.uint8)

    idx = m == 1
    out[idx] = (alpha * c + (1.0 - alpha) * out[idx]).astype(np.uint8)
    return out

def to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convenience: BGR → RGB."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convenience: RGB → BGR."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
