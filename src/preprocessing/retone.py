import numpy as np

def retone_to_target(
    v_uint8: np.ndarray,
    target_mean: float = 145.0,
    target_std: float = 80.0,
    outer_clip_lo: float = 2.0,
    outer_clip_hi: float = 98.0
) -> np.ndarray:
    """
    Retone a single-channel uint8 image towards target mean/std with percentile clipping.
    """
    v = v_uint8.astype(np.float32)

    lo = np.percentile(v, outer_clip_lo)
    hi = np.percentile(v, outer_clip_hi)
    if hi - lo > 1e-6:
        v = np.clip((v - lo) / (hi - lo), 0, 1)
    else:
        v = np.clip(v / 255.0, 0, 1)

    mu = float(v.mean())
    sd = float(v.std()) + 1e-6

    a = (target_std / 255.0) / sd
    b = (target_mean / 255.0) - a * mu
    v = np.clip(a * v + b, 0, 1)
    return (v * 255.0).astype(np.uint8)
