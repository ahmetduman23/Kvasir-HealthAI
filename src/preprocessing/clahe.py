import cv2
import numpy as np
from typing import Tuple

def apply_clahe(gray_uint8: np.ndarray, clipLimit: float = 1.5, tileGridSize: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited AHE on a single-channel uint8 image.
    """
    if gray_uint8.dtype != np.uint8:
        raise ValueError("apply_clahe expects uint8 input.")
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray_uint8)
