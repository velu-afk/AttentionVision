"""Unit tests for classification metrics, latency benchmarking, and corruption functions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from attentionvision.metrics import compute_classification_metrics, measure_inference_latency
from attentionvision.models import BaselineCNN
from attentionvision.robustness import apply_gaussian_blur, apply_gaussian_noise


def test_perfect_and_known_classification_metrics() -> None:
    """Verify accuracy, macro-F1, per-class metrics, and confusion matrix dimensions."""
    y_true = np.arange(10).repeat(3)
    y_pred = y_true.copy()

    res = compute_classification_metrics(y_true, y_pred)
    assert res["accuracy"] == pytest.approx(1.0)
    assert res["macro_precision"] == pytest.approx(1.0)
    assert res["macro_recall"] == pytest.approx(1.0)
    assert res["macro_f1"] == pytest.approx(1.0)
    assert len(res["per_class"]) == 10

    cm = np.asarray(res["confusion_matrix"])
    assert cm.shape == (10, 10)
    assert np.all(np.diag(cm) == 3)
    assert cm.sum() == 30


def test_metrics_invalid_inputs_raise_value_error() -> None:
    """Verify that empty arrays, mismatched lengths, or out-of-range labels raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        _ = compute_classification_metrics([], [])
    with pytest.raises(ValueError, match="Length mismatch"):
        _ = compute_classification_metrics([0, 1], [0])
    with pytest.raises(ValueError, match="Class indices must lie"):
        _ = compute_classification_metrics([0, 12], [0, 1])


def test_inference_latency_measurement_protocol() -> None:
    """Verify that latency benchmarking returns positive timings and metadata."""
    model = BaselineCNN()
    stats = measure_inference_latency(
        model=model,
        device=torch.device("cpu"),
        batch_size=2,
        warmup_runs=1,
        measure_runs=3,
    )
    assert stats["batch_size"] == 2
    assert stats["mean_batch_ms"] > 0.0
    assert stats["mean_sample_ms"] > 0.0
    assert stats["throughput_samples_per_sec"] > 0.0


def test_robustness_corruptions_preserve_shape_and_determinism() -> None:
    """Verify Gaussian noise and Gaussian blur corruptions on synthetic tensors."""
    x = torch.randn(4, 3, 32, 32)
    clean_noise = apply_gaussian_noise(x, sigma=0.0)
    assert torch.allclose(x, clean_noise)

    n1 = apply_gaussian_noise(x, sigma=0.15, seed=1337, batch_idx=2)
    n2 = apply_gaussian_noise(x, sigma=0.15, seed=1337, batch_idx=2)
    assert n1.shape == x.shape
    assert torch.allclose(n1, n2)
    assert not torch.allclose(x, n1)

    b1 = apply_gaussian_blur(x, sigma=1.0, kernel_size=5)
    assert b1.shape == x.shape
    assert bool(torch.isfinite(b1).all())
    assert not torch.allclose(x, b1)
