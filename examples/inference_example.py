"""
Inference script for U-Net polyp segmentation.

Usage:
  python examples/inference_example.py --img <path_or_folder> [--thr 0.5] [--size 256]
"""

import os, glob, argparse
import numpy as np
import cv2
import torch

from src.models.unet import UNet
from src.preprocessing.pipeline import preprocess_staged_rgb_single
from src.dataset.kvasir_dataset import list_pairs  # not required for single image, kept for parity

MODEL_NAME = "unet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_BASE = 32  # adjust if checkpoint used a different base

def _paths_for_unet():
    d = os.path.join("assets", "results", MODEL_NAME)
    os.makedirs(d, exist_ok=True)
    return d, os.path.join(d, f"best_{MODEL_NAME}.pt")

def build_unet(base: int | None = None) -> torch.nn.Module:
    b = UNET_BASE if base is None else int(base)
    return UNet(in_ch=3, out_ch=1, base=b)

def load_best_unet(build_fn):
    out_dir, best_pth = _paths_for_unet()
    if not os.path.exists(best_pth):
        legacy = os.path.join(out_dir, "best_unet.pt")
        if os.path.exists(legacy):
            best_pth = legacy
        else:
            raise FileNotFoundError(f"No checkpoint found under {out_dir}")
    state = torch.load(best_pth, map_location="cpu")
    model = build_fn(); model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()

def _ensure_list(path_or_dir: str):
    if os.path.isdir(path_or_dir):
        files = []
        for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
            files.extend(glob.glob(os.path.join(path_or_dir, e)))
        files.sort()
        return files
    return [path_or_dir]

def _overlay(bgr: np.ndarray, mask_u8: np.ndarray, color=(0,0,255), alpha=0.6):
    out = bgr.copy(); idx = mask_u8 > 0
    out[idx] = (alpha*np.array(color) + (1-alpha)*out[idx]).astype(np.uint8)
    return out

def infer_image(model, img_path: str, img_size: int = 256, thr: float = 0.5,
                save_dir: str = "assets/results/infer"):
    os.makedirs(save_dir, exist_ok=True)
    bgr = cv2.imread(img_path)
    if bgr is None: raise FileNotFoundError(f"Cannot read: {img_path}")

    bgr_p = preprocess_staged_rgb_single(bgr)
    rgb = cv2.cvtColor(bgr_p, cv2.COLOR_BGR2RGB)

    rgb_resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = (rgb_resized.astype(np.float32)/255.0).transpose(2,0,1)[None, ...]
    x = torch.from_numpy(x).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0,0].cpu().numpy()

    pred = (probs > thr).astype(np.uint8)*255
    pred_orig = cv2.resize(pred, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = _overlay(bgr, pred_orig)
    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(save_dir, f"{base}_mask.png"), pred_orig)
    cv2.imwrite(os.path.join(save_dir, f"{base}_overlay.png"), overlay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to an image or a folder")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--save_dir", type=str, default="assets/results/infer")
    args = ap.parse_args()

    model = load_best_unet(build_unet)
    for p in _ensure_list(args.img):
        infer_image(model, p, img_size=args.size, thr=args.thr, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
