# src/dataset/__init__.py
"""
Dataset package for Kvasir-SEG or other medical segmentation datasets.

Includes:
- list_pairs: utility to collect (image, mask) file pairs
- KvasirSegDataset: PyTorch Dataset class with preprocessing and augmentation
"""

from .kvasir_dataset import list_pairs, KvasirSegDataset

__all__ = ["list_pairs", "KvasirSegDataset"]
