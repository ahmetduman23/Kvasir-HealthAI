# 🔍 Explainable AI (XAI) — Grad-CAM for U-Net

## Overview
Explainability is achieved using **SegGradCAM**, an adaptation of Grad-CAM for segmentation models.  
It highlights the image regions that most influence the model’s prediction, allowing for interpretability and trust in medical AI systems.

---

## Method
1. A **target convolution layer** (typically the last decoder block) is selected:
   ```python
   target = model.dec4.conv[3]
2. Gradients are computed with respect to this layer’s activations.

3. These gradients are aggregated to produce a Class Activation Map (CAM).

4. The CAM is upsampled and overlaid on the original image to visualize attention regions.

## Insights

- The model primarily attends to high-contrast and well-illuminated polyp regions.

- Strong boundary activation indicates that the U-Net learns structural edges effectively.

- Slight attention diffusion occurs near irregular or blurry edges, suggesting possible improvement via attention-based modules.

## Visualization Example
| Input | Grad-CAM | Overlay |
|:------:|:---------:|:--------:|
| ![](../assets/xai_visualizations/unet_cam_00_input.png) | ![](../assets/xai_visualizations/unet_cam_00_cam.png) | ![](../assets/xai_visualizations/unet_cam_00_overlay.png) |



## Implementation Reference

- The Grad-CAM functionality is implemented under:
src/explainability/gradcam.py

## Example Usage
gradcam_demo_unet(
    model=unet,
    loader=val_loader,
    n_samples=5,
    save_dir="assets/results/xai_visualizations"
)
