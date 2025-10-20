import cv2
import numpy as np
from typing import Tuple

def remove_specular_highlight_safe(
    bgr: np.ndarray,
    s_thr: int = 60,
    v_thr: int = 230,
    achr_delta: int = 25,
    max_area_ratio: float = 0.003,
    dilate_iter: int = 1,
    inpaint_radius: int = 2,
    inpaint_method: int = cv2.INPAINT_NS,
    area_guard_ratio: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conservative specular highlight removal with area guard.
    Returns (inpainted_bgr, safe_mask).
    """
    h, w = bgr.shape[:2]
    area = h * w

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    cand = (S < s_thr) & (V > v_thr)

    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    achr = (np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)) < achr_delta

    mask = (cand & achr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=dilate_iter)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    safe_mask = np.zeros_like(mask)
    max_area = max_area_ratio * area
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            safe_mask[labels == i] = 255

    if safe_mask.sum() > area_guard_ratio * 255 * area:
        # too much area would be removed → bail out
        return bgr.copy(), np.zeros_like(safe_mask)

    out = cv2.inpaint(bgr, safe_mask, inpaintRadius=inpaint_radius, flags=inpaint_method)
    return out, safe_mask
