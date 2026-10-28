"""Unit tests for BaselineCNN and AttentionCNN architectures."""

from __future__ import annotations

import pytest
import torch

from attentionvision.config import ModelConfig
from attentionvision.models import AttentionCNN, BaselineCNN, build_model
from attentionvision.utils import count_parameters


@pytest.mark.parametrize("model_name", ["baseline_cnn", "attention_cnn"])
def test_model_forward_shape_and_finite_outputs(model_name: str) -> None:
    """Verify that both models output finite (B, 10) logits on CPU for 32x32 RGB inputs."""
    cfg = ModelConfig(name=model_name, num_classes=10)
    model = build_model(cfg)
    model.eval()

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (4, 10)
    assert bool(torch.isfinite(logits).all())


@pytest.mark.parametrize("num_classes", [2, 10, 20])
def test_configurable_num_classes(num_classes: int) -> None:
    """Verify that the classification head adapts to different class counts."""
    base = BaselineCNN(num_classes=num_classes)
    attn = AttentionCNN(num_classes=num_classes)
    x = torch.randn(2, 3, 32, 32)

    assert base(x).shape == (2, num_classes)
    assert attn(x).shape == (2, num_classes)


def test_parameter_count_overhead_is_lightweight() -> None:
    """Verify that AttentionCNN adds only the expected 2 * k^2 parameters per stage."""
    base = BaselineCNN(channels=(32, 64, 128), num_classes=10)
    attn = AttentionCNN(
        channels=(32, 64, 128),
        num_classes=10,
        attention_kernel_size=7,
        attention_stages=(2, 3),
    )
    base_params = count_parameters(base)
    attn_params = count_parameters(attn)

    # Each 7x7 SpatialAttention module without bias has 2 * 1 * 7 * 7 = 98 parameters.
    # Two stages (Stage 2 and Stage 3) add exactly 196 parameters.
    assert attn_params - base_params == 196
    assert base_params == 94762
    assert attn_params == 94958


def test_invalid_input_dimensions_raise_value_error() -> None:
    """Verify informative ValueError when input tensor is not 4D (B, 3, H, W)."""
    model = BaselineCNN()
    with pytest.raises(ValueError, match="Expected 4D input"):
        _ = model(torch.randn(3, 32, 32))
    with pytest.raises(ValueError, match="Expected 4D input"):
        _ = model(torch.randn(2, 1, 32, 32))
