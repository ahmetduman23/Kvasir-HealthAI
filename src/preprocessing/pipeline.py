import cv2
from .specular_removal import remove_specular_highlight_safe
from .homomorphic_filter import homomorphic_filter_soft
from .guided_filter import guided_filter
from .clahe import apply_clahe
from .retone import retone_to_target

def preprocess_staged_rgb_single(bgr):
    """
    BGR → (HSV-V) homomorphic → (HSV-V) guided → (HSV-V) CLAHE → (HSV-V) retone → BGR
    with specular highlight removal up front.
    """
    bgr1, _ = remove_specular_highlight_safe(bgr)

    h, s, v = cv2.split(cv2.cvtColor(bgr1, cv2.COLOR_BGR2HSV))
    v = homomorphic_filter_soft(v)
    bgr2 = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    h, s, v = cv2.split(cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV))
    v = guided_filter(v)
    bgr3 = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    h, s, v = cv2.split(cv2.cvtColor(bgr3, cv2.COLOR_BGR2HSV))
    v = apply_clahe(v)
    bgr4 = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    h, s, v = cv2.split(cv2.cvtColor(bgr4, cv2.COLOR_BGR2HSV))
    v = retone_to_target(v)
    bgr5 = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    return bgr5
