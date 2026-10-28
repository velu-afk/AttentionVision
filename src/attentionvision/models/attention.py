"""Lightweight Spatial Attention Module for CNN feature map modulation.

Mathematical Formulation:
-------------------------
Given an intermediate convolutional feature map F in R^{B x C x H x W}:

1. Channel-wise Spatial Pooling:
   Compute average pooling and max pooling across the channel axis (dim=1):
     F_avg(b, 1, h, w) = (1 / C) * sum_{c=1}^{C} F(b, c, h, w)    in R^{B x 1 x H x W}
     F_max(b, 1, h, w) = max_{c in 1..C} F(b, c, h, w)            in R^{B x 1 x H x W}

2. Spatial Descriptor Concatenation:
   Concatenate along the channel dimension:
     F_cat = [F_avg ; F_max]                                      in R^{B x 2 x H x W}

3. Spatial Context Convolution & Sigmoid Normalization:
   Apply a 2D convolution with 2 input channels, 1 output channel, odd kernel size k x k,
   and symmetric zero-padding p = (k - 1) // 2 so spatial dimensions H x W are preserved:
     Z = Conv2d_{2 -> 1, k x k}(F_cat)                            in R^{B x 1 x H x W}
     M_s(F) = sigma(Z) = 1 / (1 + exp(-Z))                        in (0, 1)^{B x 1 x H x W}

4. Multiplicative or Residual Feature Modulation (with Broadcasting):
   The spatial attention map M_s(F) of shape (B, 1, H, W) is broadcast across all C channels:
   - Standard multiplicative gating (residual=False):
       F_out = F * M_s(F)                                         in R^{B x C x H x W}
   - Residual multiplicative gating (residual=True, default):
       F_out = F * (1 + M_s(F))                                   in R^{B x C x H x W}

   Why residual gating is helpful:
   In standard multiplicative gating (F * M_s), sigmoid values near 0.5 at initialization
   attenuate activations and gradients by 50% at each stage. Residual gating F * (1 + M_s)
   preserves the identity path F while allowing the network to amplify informative spatial
   locations by up to 2x.

Important Epistemic Note:
   Spatial attention weights M_s(F) reflect learned feature-reweighting coefficients, NOT
   proven causal explanations of model predictions.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Channel-pooled spatial attention module inspired by CBAM (Woo et al., ECCV 2018).

    Attributes:
        kernel_size: Odd integer spatial convolution kernel size (e.g., 3, 5, 7).
        residual: If True, computes F * (1 + M_s(F)); if False, computes F * M_s(F).
        conv: 2-channel to 1-channel spatial convolution layer.
        sigmoid: Logistic sigmoid activation mapping logits to (0, 1).
        last_attention_map: Detached copy of the most recently computed (B, 1, H, W) map
            for inspection and visualization.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        residual: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"SpatialAttention kernel_size must be a positive odd integer, got {kernel_size}."
            )
        self.kernel_size = int(kernel_size)
        self.residual = bool(residual)
        padding = (self.kernel_size - 1) // 2

        # 2 input channels (channel-wise mean + channel-wise max) -> 1 spatial attention map
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )
        self.sigmoid = nn.Sigmoid()
        self.last_attention_map: Optional[torch.Tensor] = None

    def compute_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the normalized spatial attention map M_s(x) of shape (B, 1, H, W).

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            Attention map tensor of shape (B, 1, H, W) with values in (0, 1).
        """
        if x.ndim != 4:
            raise ValueError(
                f"SpatialAttention expects a 4D input tensor (B, C, H, W), got shape {tuple(x.shape)}."
            )
        # Channel-wise average pooling: (B, C, H, W) -> (B, 1, H, W)
        avg_out = x.mean(dim=1, keepdim=True)
        # Channel-wise max pooling: (B, C, H, W) -> (B, 1, H, W)
        max_out, _ = x.max(dim=1, keepdim=True)

        # Concatenate descriptors along channel axis: (B, 2, H, W)
        spatial_descriptor = torch.cat([avg_out, max_out], dim=1)

        # Spatial convolution + sigmoid normalization: (B, 2, H, W) -> (B, 1, H, W)
        attention_logits = self.conv(spatial_descriptor)
        attention_map = self.sigmoid(attention_logits)
        return attention_map

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply spatial attention modulation to input feature map x.

        Args:
            x: Input feature map of shape (B, C, H, W).
            return_attention: If True, returns (modulated_features, attention_map).

        Returns:
            Modulated feature map of shape (B, C, H, W), and optionally the
            attention map of shape (B, 1, H, W).
        """
        attention_map = self.compute_attention_map(x)
        self.last_attention_map = attention_map.detach()

        # Broadcast (B, 1, H, W) across (B, C, H, W)
        if self.residual:
            out = x * (1.0 + attention_map)
        else:
            out = x * attention_map

        if return_attention:
            return out, attention_map
        return out
