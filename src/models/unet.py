# src/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ConvBlock(nn.Module):
    """(Conv → BN → ReLU) × 2 block."""
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for binary segmentation.
    Returns logits (apply sigmoid outside).

    Args:
        in_ch:  input channels (RGB=3)
        out_ch: output channels (binary mask=1)
        base:   base number of feature maps (32/64/…)
        p_drop: dropout prob inside ConvBlocks (default 0.0)
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32, p_drop: float = 0.0) -> None:
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch,      base,   p_drop)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base,       base*2, p_drop)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base*2,     base*4, p_drop)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base*4,     base*8, p_drop)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base*8, base*16, p_drop)

        # Decoder (concat sonrası kanal sayısına dikkat)
        self.up4  = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ConvBlock(base*16,  base*8, p_drop)   # 8 + 8 = 16

        self.up3  = nn.ConvTranspose2d(base*8,  base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8,   base*4, p_drop)   # 4 + 4 = 8

        self.up2  = nn.ConvTranspose2d(base*4,  base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4,   base*2, p_drop)   # 2 + 2 = 4

        self.up1  = nn.ConvTranspose2d(base*2,  base,   2, stride=2)
        self.dec1 = ConvBlock(base*2,   base,   p_drop)   # 1 + 1 = 2

        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b);  d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)

        logits = self.out_conv(d1)  # (N, out_ch, H, W)
        return logits
