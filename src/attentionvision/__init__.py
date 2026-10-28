"""AttentionVision: Investigating Spatial Attention in CNN-Based Image Classification."""

from __future__ import annotations

__version__ = "0.1.0"

from attentionvision.config import ExperimentConfig, load_config
from attentionvision.models import AttentionCNN, BaselineCNN, SpatialAttention, build_model

__all__ = [
    "__version__",
    "ExperimentConfig",
    "load_config",
    "BaselineCNN",
    "AttentionCNN",
    "SpatialAttention",
    "build_model",
]
