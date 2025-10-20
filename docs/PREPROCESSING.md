# 🧼 Preprocessing

Before training, each image is enhanced using a 5-step preprocessing pipeline to improve contrast and reduce glare.

## Steps
1. **Specular Highlight Removal** – removes light reflections.  
2. **Homomorphic Filter** – normalizes illumination and enhances contrast.  
3. **Guided Filter** – smooths noise while keeping edges.  
4. **CLAHE** – improves local contrast and texture detail.  
5. **Retone Adjustment** – balances color tone and brightness.

## Implementation
Applied automatically during dataset loading:

```python
bgr = preprocess_staged_rgb_single(bgr)

Example 

| Original | After Preprocessing |
|-----------|--------------------|
| ![](../assets/images/input_01.jpg) | ![](../assets/images/preprocessed_01.jpg) |


## Effect on Training

- **Higher Dice and IoU scores**  
- **Fewer false negatives** in low-contrast areas  
- **Faster convergence** and more stable learning
