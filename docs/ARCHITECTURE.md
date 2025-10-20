# Model Architecture — U-Net for Polyp Segmentation

## Overview
This project implements a **U-Net-based encoder–decoder architecture** for semantic segmentation of colonoscopy images from the **Kvasir-SEG** dataset.  
The model is designed to detect and segment polyps with high spatial precision.

---

## Architecture Design

### Encoder
- Sequential convolutional blocks with **ReLU** activations and **Batch Normalization**.  
- Each block doubles the number of feature maps while halving the spatial resolution.  
- The encoder extracts low-to-high-level texture, color, and boundary information.

### Bottleneck
- Two convolutional layers with **1024 channels**.  
- Acts as the semantic information bridge between the encoder and decoder.

### Decoder
- Transposed convolutions for upsampling.  
- **Skip connections** concatenate encoder and decoder features to recover spatial detail.  
- Final **1×1 convolution** produces a single-channel segmentation map.

---

## Activation & Output
- **Sigmoid** activation is applied at the final layer to generate pixel-wise probabilities.  
- Thresholding at **0.5** converts the probability map into a binary mask.

---

## Training Configuration

| Parameter | Value |
|------------|-------|
| Optimizer | Adam |
| Loss | BCE + Dice Loss |
| Epochs | 80 |
| Learning Rate | 1e-3 |
| Batch Size | 2 |
| Image Size | 256×256 |
| Early Stopping | Enabled (patience = 15, min_delta = 1e-4) |
| Framework | PyTorch 2.x |

---

## Model Diagram

[Input Image]
↓
[Encoder Blocks ×4]
↓
[Bottleneck]
↓
[Decoder Blocks ×4 + Skip Connections]
↓
[1×1 Conv + Sigmoid]
↓
[Segmentation Mask]

---

## Checkpointing
- The best model is automatically saved to:
assets/results/unet/best_unet.pt

- Early stopping prevents overfitting and accelerates convergence.  
- Training history and metrics are also logged under:
assets/results/unet/history.json