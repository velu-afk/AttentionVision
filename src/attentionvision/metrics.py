"""Evaluation metrics and inference latency benchmarking for image classification."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn

from attentionvision.data import CIFAR10_CLASSES


def compute_classification_metrics(
    y_true: Union[Sequence[int], np.ndarray],
    y_pred: Union[Sequence[int], np.ndarray],
    class_names: Sequence[str] = CIFAR10_CLASSES,
) -> Dict[str, Any]:
    """Compute overall accuracy, macro precision/recall/F1, per-class metrics, and confusion matrix.

    Args:
        y_true: 1D ground-truth class labels in [0, num_classes - 1].
        y_pred: 1D predicted class labels in [0, num_classes - 1].
        class_names: Ordered sequence of class names.

    Returns:
        Dictionary containing scalar macro metrics, per-class metric records, and confusion matrix.
    """
    true_arr = np.asarray(y_true, dtype=np.int64)
    pred_arr = np.asarray(y_pred, dtype=np.int64)

    if true_arr.ndim != 1 or pred_arr.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1D arrays, got shapes {true_arr.shape} and {pred_arr.shape}."
        )
    if true_arr.size == 0 or pred_arr.size == 0:
        raise ValueError("Cannot compute classification metrics on empty arrays.")
    if true_arr.shape[0] != pred_arr.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {true_arr.shape[0]} elements, y_pred has {pred_arr.shape[0]}."
        )

    num_classes = len(class_names)
    if np.any((true_arr < 0) | (true_arr >= num_classes)) or np.any(
        (pred_arr < 0) | (pred_arr >= num_classes)
    ):
        raise ValueError(f"Class indices must lie in [0, {num_classes - 1}].")

    labels = list(range(num_classes))
    acc = float(accuracy_score(true_arr, pred_arr))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        true_arr,
        pred_arr,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        true_arr,
        pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(true_arr, pred_arr, labels=labels)

    per_class: List[Dict[str, Any]] = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": idx,
                "class_name": str(name),
                "precision": round(float(per_p[idx]), 6),
                "recall": round(float(per_r[idx]), 6),
                "f1": round(float(per_f1[idx]), 6),
                "support": int(per_sup[idx]),
            }
        )

    return {
        "num_samples": int(true_arr.shape[0]),
        "accuracy": round(acc, 6),
        "macro_precision": round(float(macro_p), 6),
        "macro_recall": round(float(macro_r), 6),
        "macro_f1": round(float(macro_f1), 6),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    in_channels: int = 3,
    image_size: int = 32,
    warmup_runs: int = 10,
    measure_runs: int = 50,
) -> Dict[str, Any]:
    """Measure forward-pass inference latency under a documented, controlled protocol.

    Protocol:
        1. Set model to eval() mode and disable gradient tracking (`torch.no_grad()`).
        2. Allocate a synthetic input batch of shape `(batch_size, in_channels, image_size, image_size)`
           on the target device.
        3. Execute `warmup_runs` forward passes (synchronizing CUDA if applicable) to warm up caches.
        4. Time `measure_runs` consecutive forward passes using `time.perf_counter()`.
        5. Report mean/std batch latency (ms), per-sample latency (ms), and throughput (samples/sec).
    """
    if batch_size <= 0 or warmup_runs < 0 or measure_runs <= 0:
        raise ValueError("Invalid latency benchmark parameters.")

    model.eval()
    model.to(device)
    dummy_input = torch.randn(batch_size, in_channels, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_times_ms: List[float] = []
        for _ in range(measure_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            batch_times_ms.append(elapsed_ms)

    times_arr = np.asarray(batch_times_ms, dtype=np.float64)
    mean_batch_ms = float(times_arr.mean())
    std_batch_ms = float(times_arr.std())
    mean_sample_ms = mean_batch_ms / float(batch_size)
    throughput = (1000.0 / mean_sample_ms) if mean_sample_ms > 0 else 0.0

    return {
        "device": str(device),
        "batch_size": int(batch_size),
        "input_shape": [int(batch_size), int(in_channels), int(image_size), int(image_size)],
        "warmup_runs": int(warmup_runs),
        "measure_runs": int(measure_runs),
        "mean_batch_ms": round(mean_batch_ms, 4),
        "std_batch_ms": round(std_batch_ms, 4),
        "mean_sample_ms": round(mean_sample_ms, 4),
        "throughput_samples_per_sec": round(throughput, 2),
    }
