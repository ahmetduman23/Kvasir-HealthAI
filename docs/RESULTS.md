# Results

## Experimental Setup
- Input size: 256×256
- Optimizer: Adam, LR=1e-3
- Loss: BCE + Dice
- Metrics: Dice, IoU

## Validation Metrics
| Epoch | Dice | IoU |
|------:|-----:|----:|
|      |      |     |

## Qualitative Samples
See `assets/results/` for predictions (input, GT, prediction).

## Discussion
- Illumination normalization improved small polyp boundary detection.
- Remaining failure cases: low-contrast tiny polyps; propose data augment & larger backbones.
