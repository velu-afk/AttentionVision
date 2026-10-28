"""Baseline CNN (Model A) and Spatial-Attention CNN (Model B) for CIFAR-10 Classification.

Both architectures share an identical 3-stage convolutional backbone so that any difference
in accuracy, calibration, corruption sensitivity, or latency is attributable strictly to the
spatial-attention modules rather than confounding backbone changes.

Stage-by-Stage Tensor Shapes (for CIFAR-10 input of shape (B, 3, 32, 32)):
--------------------------------------------------------------------------
- Input Image:                 (B,   3, 32, 32)
- Stage 1 ConvBlock:
    Conv2d(3 -> 32, 3x3, p=1)  (B,  32, 32, 32)
    BatchNorm2d(32) + ReLU     (B,  32, 32, 32)
    [Optional Attention 1]     (B,  32, 32, 32)  (Attention map: (B, 1, 32, 32))
    MaxPool2d(2x2, stride=2)   (B,  32, 16, 16)
- Stage 2 ConvBlock:
    Conv2d(32 -> 64, 3x3, p=1) (B,  64, 16, 16)
    BatchNorm2d(64) + ReLU     (B,  64, 16, 16)
    [Attention 2 (default)]    (B,  64, 16, 16)  (Attention map: (B, 1, 16, 16))
    MaxPool2d(2x2, stride=2)   (B,  64,  8,  8)
- Stage 3 ConvBlock:
    Conv2d(64 -> 128, 3x3,p=1) (B, 128,  8,  8)
    BatchNorm2d(128) + ReLU    (B, 128,  8,  8)
    [Attention 3 (default)]    (B, 128,  8,  8)  (Attention map: (B, 1,  8,  8))
    MaxPool2d(2x2, stride=2)   (B, 128,  4,  4)
- Global Head:
    AdaptiveAvgPool2d(1x1)     (B, 128,  1,  1)
    Flatten                    (B, 128)
    Dropout(p=dropout_rate)    (B, 128)
    Linear(128 -> 10)          (B,  10)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from attentionvision.config import ModelConfig
from attentionvision.models.attention import SpatialAttention


class ConvBlock(nn.Module):
    """Single convolutional stage: Conv2d (3x3, pad=1) -> BatchNorm2d -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Bias is omitted in Conv2d because BatchNorm2d immediately subtracts the channel mean
        # and learns its own shift parameter (beta).
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BaselineCNN(nn.Module):
    """Model A: Compact 3-stage Convolutional Neural Network baseline for CIFAR-10."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        num_classes: int = 10,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"Expected 3 stage channel sizes, got {channels}.")
        c1, c2, c3 = int(channels[0]), int(channels[1]), int(channels[2])

        self.in_channels = int(in_channels)
        self.channels = (c1, c2, c3)
        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout_rate)

        # Stage 1: (B, 3, 32, 32) -> (B, c1, 32, 32) -> pool -> (B, c1, 16, 16)
        self.block1 = ConvBlock(self.in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: (B, c1, 16, 16) -> (B, c2, 16, 16) -> pool -> (B, c2, 8, 8)
        self.block2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: (B, c2, 8, 8) -> (B, c3, 8, 8) -> pool -> (B, c3, 4, 4)
        self.block3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification head: (B, c3, 4, 4) -> (B, c3, 1, 1) -> (B, c3) -> (B, num_classes)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(c3, self.num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input tensor through the 3 convolutional stages."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected 4D input (B, {self.in_channels}, H, W), got shape {tuple(x.shape)}."
            )
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits of shape (B, num_classes)."""
        features = self.forward_features(x)
        pooled = self.global_pool(features)
        flat = self.flatten(pooled)
        dropped = self.dropout(flat)
        logits = self.classifier(dropped)
        return logits


class AttentionCNN(nn.Module):
    """Model B: CNN augmented with lightweight SpatialAttention modules.

    Uses the exact same convolutional backbone and classification head as BaselineCNN,
    inserting SpatialAttention modules before spatial pooling at configured stages
    (default: Stage 2 at 16x16 resolution and Stage 3 at 8x8 resolution).
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        num_classes: int = 10,
        dropout_rate: float = 0.25,
        attention_kernel_size: int = 7,
        attention_residual: bool = True,
        attention_stages: Sequence[int] = (2, 3),
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"Expected 3 stage channel sizes, got {channels}.")
        c1, c2, c3 = int(channels[0]), int(channels[1]), int(channels[2])

        self.in_channels = int(in_channels)
        self.channels = (c1, c2, c3)
        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout_rate)
        self.attention_kernel_size = int(attention_kernel_size)
        self.attention_residual = bool(attention_residual)
        self.attention_stages = tuple(sorted(set(int(s) for s in attention_stages)))

        # Identical backbone stages to BaselineCNN
        self.block1 = ConvBlock(self.in_channels, c1)
        self.attn1: Optional[SpatialAttention] = (
            SpatialAttention(kernel_size=self.attention_kernel_size, residual=self.attention_residual)
            if 1 in self.attention_stages
            else None
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = ConvBlock(c1, c2)
        self.attn2: Optional[SpatialAttention] = (
            SpatialAttention(kernel_size=self.attention_kernel_size, residual=self.attention_residual)
            if 2 in self.attention_stages
            else None
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = ConvBlock(c2, c3)
        self.attn3: Optional[SpatialAttention] = (
            SpatialAttention(kernel_size=self.attention_kernel_size, residual=self.attention_residual)
            if 3 in self.attention_stages
            else None
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(c3, self.num_classes)

    def forward_with_attention(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute logits and return a dictionary of intermediate spatial attention maps."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected 4D input (B, {self.in_channels}, H, W), got shape {tuple(x.shape)}."
            )
        attn_maps: Dict[str, torch.Tensor] = {}

        x = self.block1(x)
        if self.attn1 is not None:
            x, m1 = self.attn1(x, return_attention=True)
            attn_maps["stage1"] = m1
        x = self.pool1(x)

        x = self.block2(x)
        if self.attn2 is not None:
            x, m2 = self.attn2(x, return_attention=True)
            attn_maps["stage2"] = m2
        x = self.pool2(x)

        x = self.block3(x)
        if self.attn3 is not None:
            x, m3 = self.attn3(x, return_attention=True)
            attn_maps["stage3"] = m3
        x = self.pool3(x)

        pooled = self.global_pool(x)
        flat = self.flatten(pooled)
        dropped = self.dropout(flat)
        logits = self.classifier(dropped)
        return logits, attn_maps

    def get_last_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Return the most recently cached attention maps from active stages."""
        maps: Dict[str, torch.Tensor] = {}
        for idx, mod in ((1, self.attn1), (2, self.attn2), (3, self.attn3)):
            if mod is not None and mod.last_attention_map is not None:
                maps[f"stage{idx}"] = mod.last_attention_map
        return maps

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute class logits of shape (B, num_classes)."""
        logits, attn_maps = self.forward_with_attention(x)
        if return_attention:
            return logits, attn_maps
        return logits


def build_model(model_cfg: ModelConfig) -> nn.Module:
    """Instantiate BaselineCNN or AttentionCNN from a ModelConfig."""
    name = model_cfg.name.strip().lower()
    if name == "baseline_cnn":
        return BaselineCNN(
            in_channels=model_cfg.in_channels,
            channels=model_cfg.channels,
            num_classes=model_cfg.num_classes,
            dropout_rate=model_cfg.dropout_rate,
        )
    if name == "attention_cnn":
        return AttentionCNN(
            in_channels=model_cfg.in_channels,
            channels=model_cfg.channels,
            num_classes=model_cfg.num_classes,
            dropout_rate=model_cfg.dropout_rate,
            attention_kernel_size=model_cfg.attention_kernel_size,
            attention_residual=model_cfg.attention_residual,
            attention_stages=model_cfg.attention_stages,
        )
    raise ValueError(
        f"Unknown model name '{model_cfg.name}'. Supported models: 'baseline_cnn', 'attention_cnn'."
    )
