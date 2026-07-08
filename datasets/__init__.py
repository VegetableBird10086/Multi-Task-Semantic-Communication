"""VQAv2 dataset exports.

Datasets are instantiated by the entry points, so importing this package does
not try to open every train/validation file immediately.
"""

from .vqa_mask import VQAv2, VQAv2Test

__all__ = ["VQAv2", "VQAv2Test"]
