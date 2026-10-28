"""Controlled image-corruption sensitivity experiments (Gaussian noise and Gaussian blur).

Methodological Scope Note:
--------------------------
This module evaluates model sensitivity to two synthetic pixel-space perturbations applied
only during test-time evaluation:
1. Additive zero-mean Gaussian noise in normalized tensor space with standard deviation `sigma`.
2. Spatial low-pass Gaussian blur with kernel size `k` and spatial standard deviation `sigma`.

Synthetic corruption benchmarks test specific parametric perturbations and do NOT prove
general real-world out-of-distribution (OOD) robustness. Both models are evaluated on the
exact same deterministically corrupted batches (`corruption_seed`).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attentionvision.config import ExperimentConfig, config_from_dict
from attentionvision.data import CIFAR10_CLASSES, build_dataloaders
from attentionvision.metrics import compute_classification_metrics
from attentionvision.models import build_model
from attentionvision.train import load_checkpoint
from attentionvision.utils import resolve_device, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def apply_gaussian_noise(
    images: torch.Tensor,
    sigma: float,
    seed: int = 1337,
    batch_idx: int = 0,
) -> torch.Tensor:
    """Apply deterministic additive Gaussian noise N(0, sigma^2) to a batch of images."""
    if sigma < 0.0:
        raise ValueError(f"Gaussian noise sigma must be >= 0, got {sigma}.")
    if sigma == 0.0:
        return images.clone()

    rng = np.random.default_rng(int(seed) + int(batch_idx) * 10007)
    noise_arr = rng.normal(loc=0.0, scale=float(sigma), size=tuple(images.shape)).astype(
        np.float32
    )
    noise_tensor = torch.tensor(noise_arr, dtype=images.dtype, device=images.device)
    return images + noise_tensor


def apply_gaussian_blur(
    images: torch.Tensor,
    sigma: float,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Apply spatial 2D Gaussian blur with standard deviation `sigma` to each image channel."""
    if sigma < 0.0:
        raise ValueError(f"Gaussian blur sigma must be >= 0, got {sigma}.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"blur_kernel_size must be a positive odd integer, got {kernel_size}.")
    if sigma == 0.0:
        return images.clone()

    arr = images.detach().cpu().numpy()
    radius = (kernel_size - 1) // 2
    truncate = max(1.0, float(radius) / float(sigma))
    # Apply 2D filter across spatial axes (-2, -1) only
    blurred = gaussian_filter(arr, sigma=(0.0, 0.0, float(sigma), float(sigma)), truncate=truncate)
    return torch.tensor(blurred.astype(np.float32), dtype=images.dtype, device=images.device)


def evaluate_under_corruption(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corruption_type: str,
    sigma: float,
    corruption_seed: int = 1337,
    blur_kernel_size: int = 5,
    class_names: Sequence[str] = CIFAR10_CLASSES,
) -> Dict[str, Any]:
    """Evaluate `model` on `loader` under a specific corruption type and severity `sigma`."""
    if corruption_type not in ("clean", "gaussian_noise", "gaussian_blur"):
        raise ValueError(
            f"Unsupported corruption_type '{corruption_type}'. "
            "Expected 'clean', 'gaussian_noise', or 'gaussian_blur'."
        )

    model.eval()
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if corruption_type == "clean" or sigma == 0.0:
                corrupted = images
            elif corruption_type == "gaussian_noise":
                corrupted = apply_gaussian_noise(
                    images, sigma=sigma, seed=corruption_seed, batch_idx=batch_idx
                )
            else:
                corrupted = apply_gaussian_blur(
                    images, sigma=sigma, kernel_size=blur_kernel_size
                )

            corrupted = corrupted.to(device)
            logits = model(corrupted)
            preds = logits.argmax(dim=1)

            all_true.append(labels.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    return {
        "corruption_type": corruption_type,
        "sigma": float(sigma),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "num_samples": metrics["num_samples"],
    }


def plot_robustness_curves(
    records: Sequence[Dict[str, Any]],
    corruption_type: str,
    output_path: Union[str, Path],
    title: str,
    xlabel: str,
) -> Path:
    """Plot accuracy and macro-F1 degradation curves across corruption severities."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in records if r["corruption_type"] == corruption_type]
    models = sorted(set(str(r["model_name"]) for r in filtered))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {"baseline_cnn": "#1f4e78", "attention_cnn": "#d95f02"}
    markers = {"baseline_cnn": "o", "attention_cnn": "s"}

    for m_name in models:
        m_rows = sorted(
            [r for r in filtered if r["model_name"] == m_name],
            key=lambda x: float(x["sigma"]),
        )
        sigmas = [float(r["sigma"]) for r in m_rows]
        accs = [float(r["accuracy"]) * 100.0 for r in m_rows]
        f1s = [float(r["macro_f1"]) * 100.0 for r in m_rows]
        c = colors.get(m_name, "#2b8cbe")
        mk = markers.get(m_name, "o")

        axes[0].plot(sigmas, accs, marker=mk, linewidth=2, color=c, label=m_name)
        axes[1].plot(sigmas, f1s, marker=mk, linewidth=2, color=c, label=m_name)

    axes[0].set_title("Accuracy (%) vs. Corruption Severity", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(frameon=True)

    axes[1].set_title("Macro-F1 (%) vs. Corruption Severity", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(frameon=True)

    fig.suptitle(title, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_robustness_experiment(
    cfg: ExperimentConfig,
    baseline_checkpoint: Union[str, Path],
    attention_checkpoint: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run controlled Gaussian noise and Gaussian blur evaluations on both checkpoints."""
    device = resolve_device(cfg.device)
    bundle = build_dataloaders(cfg.data, seed=cfg.seed)
    res_dir = Path(output_dir) if output_dir is not None else (Path(cfg.training.results_dir) / "robustness")
    res_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        ("baseline_cnn", baseline_checkpoint),
        ("attention_cnn", attention_checkpoint),
    ]
    records: List[Dict[str, Any]] = []

    for expected_name, ckpt_path in checkpoints:
        raw = load_checkpoint(ckpt_path, device="cpu")
        m_cfg = config_from_dict(raw["config"]).model
        model = build_model(m_cfg).to(device)
        model.load_state_dict(raw["model_state_dict"])
        model.eval()
        model_label = m_cfg.name or expected_name

        # 1. Gaussian noise sweep (includes sigma=0.0 clean reference)
        for sigma in cfg.robustness.gaussian_noise_sigmas:
            res = evaluate_under_corruption(
                model=model,
                loader=bundle.test_loader,
                device=device,
                corruption_type="gaussian_noise",
                sigma=float(sigma),
                corruption_seed=cfg.robustness.corruption_seed,
                blur_kernel_size=cfg.robustness.blur_kernel_size,
                class_names=bundle.class_names,
            )
            res["model_name"] = model_label
            res["seed"] = int(raw.get("seed", cfg.seed))
            records.append(res)

        # 2. Gaussian blur sweep (includes sigma=0.0 clean reference)
        for sigma in cfg.robustness.gaussian_blur_sigmas:
            res = evaluate_under_corruption(
                model=model,
                loader=bundle.test_loader,
                device=device,
                corruption_type="gaussian_blur",
                sigma=float(sigma),
                corruption_seed=cfg.robustness.corruption_seed,
                blur_kernel_size=cfg.robustness.blur_kernel_size,
                class_names=bundle.class_names,
            )
            res["model_name"] = model_label
            res["seed"] = int(raw.get("seed", cfg.seed))
            records.append(res)

    # Save CSV
    csv_path = res_dir / "robustness_results.csv"
    fieldnames = [
        "model_name",
        "seed",
        "corruption_type",
        "sigma",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "num_samples",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    noise_plot = plot_robustness_curves(
        records=records,
        corruption_type="gaussian_noise",
        output_path=res_dir / "robustness_gaussian_noise.png",
        title="Sensitivity to Additive Gaussian Noise (Evaluation-Only Perturbation)",
        xlabel="Noise Standard Deviation (sigma)",
    )
    blur_plot = plot_robustness_curves(
        records=records,
        corruption_type="gaussian_blur",
        output_path=res_dir / "robustness_gaussian_blur.png",
        title="Sensitivity to Spatial Gaussian Blur (Evaluation-Only Perturbation)",
        xlabel="Blur Sigma (pixels)",
    )

    summary = {
        "experiment_name": cfg.experiment_name,
        "run_type": cfg.run_type,
        "is_synthetic_data": bundle.is_synthetic,
        "corruption_seed": cfg.robustness.corruption_seed,
        "blur_kernel_size": cfg.robustness.blur_kernel_size,
        "gaussian_noise_sigmas": cfg.robustness.gaussian_noise_sigmas,
        "gaussian_blur_sigmas": cfg.robustness.gaussian_blur_sigmas,
        "csv_path": str(csv_path),
        "noise_plot": str(noise_plot),
        "blur_plot": str(blur_plot),
        "records": records,
    }
    save_json(summary, res_dir / "robustness_summary.json")
    return summary
