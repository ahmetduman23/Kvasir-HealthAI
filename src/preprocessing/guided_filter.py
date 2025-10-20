import cv2
import numpy as np
from typing import Optional

def guided_filter(src: np.ndarray, guidance: Optional[np.ndarray] = None, r: int = 6, eps: float = 0.02**2) -> np.ndarray:
    """
    Edge-preserving smoothing. src/guidance are uint8 single-channel images.
    """
    I = (guidance if guidance is not None else src).astype(np.float32) / 255.0
    p = src.astype(np.float32) / 255.0

    k = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(I, -1, k, normalize=True)
    mean_p = cv2.boxFilter(p, -1, k, normalize=True)
    corr_I  = cv2.boxFilter(I * I, -1, k, normalize=True)
    corr_Ip = cv2.boxFilter(I * p, -1, k, normalize=True)

    var_I  = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, k, normalize=True)
    mean_b = cv2.boxFilter(b, -1, k, normalize=True)
    q = mean_a * I + mean_b

    q = np.clip(q * 255.0, 0, 255)
    return q.astype(np.uint8)
