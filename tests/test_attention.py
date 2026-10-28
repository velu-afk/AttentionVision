"""Unit tests for the SpatialAttention module and AttentionCNN attention maps."""

from __future__ import annotations

import pytest
import torch

from attentionvision.models import AttentionCNN, SpatialAttention


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_spatial_attention_shapes_and_bounds(kernel_size: int) -> None:
    """Verify that SpatialAttention preserves feature shape (B, C, H, W) and bounds map in (0, 1)."""
    attn = SpatialAttention(kernel_size=kernel_size, residual=True)
    x = torch.randn(4, 64, 16, 16)

    out, amap = attn(x, return_attention=True)
    assert out.shape == (4, 64, 16, 16)
    assert amap.shape == (4, 1, 16, 16)
    assert bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(amap).all())
    assert float(amap.min().item()) >= 0.0
    assert float(amap.max().item()) <= 1.0


def test_spatial_attention_residual_vs_multiplicative_formula() -> None:
    """Verify exact mathematical behavior of residual=True vs residual=False."""
    x = torch.randn(2, 16, 8, 8)
    attn_res = SpatialAttention(kernel_size=3, residual=True)
    attn_mul = SpatialAttention(kernel_size=3, residual=False)
    # Copy conv weights so both modules produce the exact same attention map M_s(x)
    attn_mul.load_state_dict(attn_res.state_dict())

    out_res, map_res = attn_res(x, return_attention=True)
    out_mul, map_mul = attn_mul(x, return_attention=True)

    assert torch.allclose(map_res, map_mul, atol=1e-6)
    assert torch.allclose(out_mul, x * map_mul, atol=1e-6)
    assert torch.allclose(out_res, x * (1.0 + map_res), atol=1e-6)


def test_spatial_attention_gradient_propagation() -> None:
    """Verify that gradients flow through SpatialAttention to both input x and conv.weight."""
    attn = SpatialAttention(kernel_size=7, residual=True)
    x = torch.randn(3, 32, 16, 16, requires_grad=True)

    out = attn(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert attn.conv.weight.grad is not None
    assert x.grad.shape == (3, 32, 16, 16)
    assert attn.conv.weight.grad.shape == (1, 2, 7, 7)
    assert bool(torch.isfinite(x.grad).all())
    assert bool(torch.isfinite(attn.conv.weight.grad).all())


def test_attention_cnn_intermediate_map_resolutions() -> None:
    """Verify that AttentionCNN returns (B, 1, 16, 16) and (B, 1, 8, 8) maps for stages 2 and 3."""
    model = AttentionCNN(attention_stages=(2, 3))
    x = torch.randn(2, 3, 32, 32)

    logits, maps = model(x, return_attention=True)
    assert logits.shape == (2, 10)
    assert set(maps.keys()) == {"stage2", "stage3"}
    assert maps["stage2"].shape == (2, 1, 16, 16)
    assert maps["stage3"].shape == (2, 1, 8, 8)


def test_invalid_kernel_size_raises_error() -> None:
    """Verify that even or non-positive kernel sizes raise ValueError."""
    with pytest.raises(ValueError, match="positive odd integer"):
        _ = SpatialAttention(kernel_size=4)
    with pytest.raises(ValueError, match="positive odd integer"):
        _ = SpatialAttention(kernel_size=0)
