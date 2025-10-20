"""
Grad-CAM demo for U-Net (SegGradCAM).

Usage:
  python examples/explainability_demo.py --img_dir D:/Kvasir-SEG/images --mask_dir D:/Kvasir-SEG/masks --n 5
"""

import os, argparse, cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.explainability.gradcam import SegGradCAM
from src.dataset.kvasir_dataset import list_pairs, KvasirSegDataset  # ← burası artık net!

MODEL_NAME = "unet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_BASE = 32

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

def pick_unet_target(model: torch.nn.Module):
    try:
        return model.dec4.conv[3]
    except Exception:
        last = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last = m
        return last

def gradcam_on_batch(model: torch.nn.Module, xb: torch.Tensor, alpha: float = 0.35):
    target = pick_unet_target(model)
    if target is None:
        raise RuntimeError("Grad-CAM target layer not found.")
    engine = SegGradCAM(model, target_layer=target, device=DEVICE)
    outs = []
    for i in range(xb.size(0)):
        x = xb[i:i+1].to(DEVICE).requires_grad_(True)
        with torch.enable_grad():
            probs, cam = engine.generate(x)
        rgb = (x[0].detach().permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        overlay_bgr = engine.overlay(rgb[:,:,::-1], cam, alpha=alpha)
        outs.append((rgb, cam, overlay_bgr[:,:,::-1]))
    engine.close()
    return outs

def demo_with_loader(model, loader, n: int = 5, save_dir: str | None = None):
    shown = 0
    for xb, _ in loader:
        for rgb, cam, overlay in gradcam_on_batch(model, xb):
            if shown >= n: return
            plt.figure(figsize=(9,3))
            plt.subplot(1,3,1); plt.imshow(rgb);            plt.title("Input");    plt.axis("off")
            plt.subplot(1,3,2); plt.imshow(cam, cmap="jet"); plt.title("Grad-CAM"); plt.axis("off")
            plt.subplot(1,3,3); plt.imshow(overlay);        plt.title("Overlay");  plt.axis("off")
            plt.tight_layout(); plt.show()

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                base = f"unet_cam_{shown:02d}"
                cv2.imwrite(os.path.join(save_dir, f"{base}_input.png"), rgb[:, :, ::-1])
                cv2.imwrite(os.path.join(save_dir, f"{base}_cam.png"),   (cam * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_dir, f"{base}_overlay.png"), overlay[:, :, ::-1])
            shown += 1

def build_val_loader(img_dir: str, mask_dir: str, img_size: int = 256, batch_size: int = 2):
    pairs = list_pairs(img_dir, mask_dir)
    n_val = max(1, int(0.2 * len(pairs)))
    va = pairs[-n_val:]
    ds = KvasirSegDataset(va, img_size=img_size, is_train=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--mask_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--save_dir", type=str, default="assets/results/xai_visualizations")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    model = load_best_unet(build_unet)
    loader = build_val_loader(args.img_dir, args.mask_dir, img_size=args.size, batch_size=2)
    demo_with_loader(model, loader, n=args.n, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
