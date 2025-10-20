import numpy as np

def homomorphic_filter_soft(
    v_uint8: np.ndarray,
    cutoff: float = 100.0,
    gamma_l: float = 0.99,
    gamma_h: float = 1.03,
    blend: float = 0.90,
    inner_clip: float = 3.0
) -> np.ndarray:
    """
    Gentle homomorphic illumination normalization on a single uint8 channel.
    """
    g0 = v_uint8.astype(np.float32) / 255.0
    log_g = np.log1p(g0)

    dft = np.fft.fft2(log_g)
    dft_shift = np.fft.fftshift(dft)

    h, w = v_uint8.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = (xx - cx) ** 2 + (yy - cy) ** 2

    H = 1.0 - np.exp(-(dist / (2.0 * (cutoff ** 2))))
    H = (gamma_h - gamma_l) * H + gamma_l

    rec = np.fft.ifft2(np.fft.ifftshift(dft_shift * H)).real
    out = np.expm1(rec)
    out = np.clip(out, 0, 1)

    # mean-preserving
    out *= (float(g0.mean() + 1e-6) / float(out.mean() + 1e-6))

    # gentle percentile clip & normalize
    if inner_clip and inner_clip > 0:
        lo = np.percentile(out, inner_clip)
        hi = np.percentile(out, 100.0 - inner_clip)
        if hi - lo > 1e-6:
            out = np.clip((out - lo) / (hi - lo), 0, 1)

    # blend with original to avoid over-correction
    out = blend * out + (1.0 - blend) * g0
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)
