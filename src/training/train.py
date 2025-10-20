# src/training/train.py
"""
Train U-Net on Kvasir-SEG with robust guards:
- lower LR + weight decay
- grad clipping
- NaN/Inf guards for loss/logits/metrics
- ReduceLROnPlateau scheduler
- early stopping + history/summary

Usage:
  python -m src.training.train --img_dir D:/Kvasir-SEG/images --mask_dir D:/Kvasir-SEG/masks
"""

import os, json, time, random, platform
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet import UNet
from src.training.losses import BCEDiceLoss
from src.dataset import list_pairs, KvasirSegDataset

MODEL_NAME = "unet"
UNET_BASE  = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def _paths_for_unet():
    d = os.path.join("assets", "results", MODEL_NAME)
    os.makedirs(d, exist_ok=True)
    return d, os.path.join(d, f"best_{MODEL_NAME}.pt")

def _save_json(obj: dict, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _has_nonfinite(t: torch.Tensor) -> bool:
    return not torch.isfinite(t).all()

def _train_one_epoch(model, loader, optimizer, criterion, max_grad_norm=1.0):
    model.train()
    total = 0.0
    for xb, yb in tqdm(loader, leave=False, desc="train"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        # (opsiyonel) tamamen boş maskeli batch'i atla
        if (yb.sum() == 0).item():
            continue

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)

        if _has_nonfinite(logits):
            print("[warn] non-finite logits; skipping batch")
            continue

        loss = criterion(logits, yb)
        if not torch.isfinite(loss):
            print("[warn] non-finite loss; skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total += float(loss.item()) * xb.size(0)
    denom = max(1, len(loader.dataset))
    return total / denom

@torch.no_grad()
def _validate(model, loader, criterion):
    model.eval()
    total = 0.0; dices = []; ious = []

    for xb, yb in tqdm(loader, leave=False, desc="valid"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)

        if _has_nonfinite(logits):
            print("[warn] non-finite logits on val; forcing large loss")
            loss = torch.tensor(1.0, device=xb.device)
        else:
            loss = criterion(logits, yb)

        total += float(loss.item()) * xb.size(0)

        probs = torch.sigmoid(logits).float()
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)

        # Soft Dice
        inter = (probs * yb).sum(dim=(1,2,3))
        denom = (probs*probs).sum(dim=(1,2,3)) + (yb*yb).sum(dim=(1,2,3)) + 1e-6
        dice = ((2*inter + 1e-6) / denom)
        dice = torch.nan_to_num(dice, nan=0.0).mean().item()
        dices.append(dice)

        # IoU (0.5 threshold)
        pred = (probs > 0.5).float()
        inter_iou = (pred * yb).sum(dim=(1,2,3))
        union_iou = (pred + yb).clamp(0,1).sum(dim=(1,2,3)) + 1e-6
        iou = (inter_iou / union_iou)
        iou = torch.nan_to_num(iou, nan=0.0).mean().item()
        ious.append(iou)

    avg_loss = total / max(1, len(loader.dataset))
    return avg_loss, float(np.mean(dices)), float(np.mean(ious))

def train(
    img_dir: str,
    mask_dir: str,
    img_size: int = 256,
    batch_size: int = 2,
    epochs: int = 80,
    lr: float = 3e-4,          # ↓ düşürüldü
    weight_decay: float = 1e-5,# ↑ eklendi
    patience: int = 15,
    min_delta: float = 1e-4,
    base: int = UNET_BASE,
):
    pairs = list_pairs(img_dir, mask_dir)
    random.shuffle(pairs)
    n_val = max(1, int(0.2 * len(pairs)))
    tr, va = pairs[:-n_val], pairs[-n_val:]

    train_loader = DataLoader(
        KvasirSegDataset(tr, img_size, is_train=True),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        KvasirSegDataset(va, img_size, is_train=False),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    model = UNet(in_ch=3, out_ch=1, base=base).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )
    criterion = BCEDiceLoss(bce_weight=0.5)

    best_dice = -np.inf
    epochs_no_improve = 0
    out_dir, best_path = _paths_for_unet()

    history = {"epoch": [], "train_loss": [], "val_loss": [], "dice": [], "iou": [], "lr": []}

    print(f"### Training {MODEL_NAME} on {DEVICE} (patience={patience}, min_delta={min_delta})")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = _train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, dice, iou = _validate(model, val_loader, criterion)
        dt = time.time() - t0

        # log
        history["epoch"].append(ep)
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(va_loss))
        history["dice"].append(float(dice))
        history["iou"].append(float(iou))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(f"[{MODEL_NAME}] Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f} "
              f"| Dice {dice:.4f} | IoU {iou:.4f} | LR {optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")

        # scheduler: val dice ile
        scheduler.step(dice)

        # early stopping + checkpoint
        if dice > best_dice + float(min_delta):
            best_dice = dice
            torch.save(model.state_dict(), best_path)
            epochs_no_improve = 0
            print(f"[{MODEL_NAME}] ➜ best updated: Dice {best_dice:.4f} → {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{MODEL_NAME}] Early stopping (no improve {patience} epochs).")
                break

    # best'i yükle ve özetleri yaz
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model = model.to(DEVICE).eval()

    summary = {
        "model": MODEL_NAME,
        "best_path": best_path if os.path.exists(best_path) else None,
        "best_dice": float(best_dice) if best_dice != -np.inf else None,
        "img_size": img_size, "batch_size": batch_size, "epochs": len(history["epoch"]),
        "lr_init": lr, "weight_decay": weight_decay,
        "device": str(DEVICE), "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "python": platform.python_version(),
        "patience": patience, "min_delta": float(min_delta),
    }
    _save_json(summary, os.path.join(out_dir, "summary.json"))
    _save_json(history, os.path.join(out_dir, "history.json"))

    print(f"[{MODEL_NAME}] Best Dice: {best_dice:.4f}" if best_dice != -np.inf else f"[{MODEL_NAME}] No improvement recorded.")
    return model, history

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, default=os.environ.get("KVASIR_IMG_DIR", "./Kvasir-SEG/images"))
    ap.add_argument("--mask_dir", type=str, default=os.environ.get("KVASIR_MSK_DIR", "./Kvasir-SEG/masks"))
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=UNET_BASE)
    args = ap.parse_args()

    train(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        base=args.base,
    )
