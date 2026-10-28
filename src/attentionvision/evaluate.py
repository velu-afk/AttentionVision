"""Model evaluation, prediction collection, metrics export, and matplotlib plotting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attentionvision.config import ExperimentConfig, config_from_dict
from attentionvision.data import CIFAR10_CLASSES, build_dataloaders, denormalize_image
from attentionvision.metrics import compute_classification_metrics, measure_inference_latency
from attentionvision.models import AttentionCNN, build_model
from attentionvision.train import load_checkpoint
from attentionvision.utils import count_parameters, resolve_device, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run inference over `loader` and return (y_true, y_pred, mean_cross_entropy_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            bs = int(labels.shape[0])
            total_loss += float(loss.item()) * bs
            total_samples += bs
            all_true.append(labels.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.array([], dtype=np.int64)
    avg_loss = total_loss / max(1, total_samples)
    return y_true, y_pred, round(avg_loss, 6)


def plot_training_curves(
    history: Sequence[Dict[str, Any]],
    output_dir: Union[str, Path],
    model_title: str = "Model",
) -> Tuple[Optional[Path], Optional[Path]]:
    """Generate training vs. validation loss and accuracy curves from saved epoch history."""
    if not history:
        return None, None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    val_loss = [float(h["val_loss"]) for h in history]
    train_acc = [float(h["train_acc"]) * 100.0 for h in history]
    val_acc = [float(h["val_acc"]) * 100.0 for h in history]

    # 1. Loss curves
    loss_path = out_dir / "loss_curves.png"
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(epochs, train_loss, marker="o", linewidth=2, color="#1f4e78", label="Train Loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2, color="#d95f02", label="Validation Loss")
    ax.set_title(f"{model_title} — Cross-Entropy Loss vs. Epoch", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Accuracy curves
    acc_path = out_dir / "accuracy_curves.png"
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(epochs, train_acc, marker="o", linewidth=2, color="#1f4e78", label="Train Accuracy (%)")
    ax.plot(epochs, val_acc, marker="s", linewidth=2, color="#2b8cbe", label="Validation Accuracy (%)")
    ax.set_title(f"{model_title} — Accuracy (%) vs. Epoch", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return loss_path, acc_path


def plot_confusion_matrix(
    cm: Union[Sequence[Sequence[int]], np.ndarray],
    class_names: Sequence[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> Path:
    """Plot and save an annotated confusion matrix heatmap."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm_arr = np.asarray(cm, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Sample Count")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    thresh = cm_arr.max() / 2.0 if cm_arr.size > 0 else 0.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                str(cm_arr[i, j]),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "#1a1a1a",
                fontsize=7.5,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_class_metrics(
    per_class: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Per-Class Precision, Recall, and F1",
) -> Path:
    """Plot grouped bar chart of per-class precision, recall, and F1 score."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = [str(item["class_name"]) for item in per_class]
    prec = [float(item["precision"]) * 100.0 for item in per_class]
    rec = [float(item["recall"]) * 100.0 for item in per_class]
    f1 = [float(item["f1"]) * 100.0 for item in per_class]

    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.bar(x - width, prec, width, label="Precision (%)", color="#1f4e78")
    ax.bar(x, rec, width, label="Recall (%)", color="#2b8cbe")
    ax.bar(x + width, f1, width, label="F1-Score (%)", color="#e66101")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("CIFAR-10 Class")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_attention_maps(
    model: AttentionCNN,
    dataset: Any,
    device: torch.device,
    output_path: Union[str, Path],
    mean: Sequence[float] = (0.4914, 0.4822, 0.4465),
    std: Sequence[float] = (0.2470, 0.2435, 0.2616),
    num_samples: int = 4,
) -> Optional[Path]:
    """Visualize input images alongside Stage 2 (16x16) and Stage 3 (8x8) spatial attention maps."""
    if not isinstance(model, AttentionCNN):
        return None
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(num_samples, len(dataset))
    if n == 0:
        return None

    model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for i in range(n):
            img_t, label = dataset[i]
            inp = img_t.unsqueeze(0).to(device)
            logits, attn_maps = model.forward_with_attention(inp)
            pred = int(logits.argmax(dim=1).item())

            rgb = denormalize_image(img_t, mean=mean, std=std)
            axes[i, 0].imshow(rgb)
            true_name = CIFAR10_CLASSES[int(label)] if int(label) < 10 else str(label)
            pred_name = CIFAR10_CLASSES[pred] if pred < 10 else str(pred)
            axes[i, 0].set_title(f"Input (T:{true_name} / P:{pred_name})", fontsize=8.5)
            axes[i, 0].axis("off")

            for col_idx, stage_key in ((1, "stage2"), (2, "stage3")):
                ax = axes[i, col_idx]
                if stage_key in attn_maps:
                    amap = attn_maps[stage_key][0, 0].detach().cpu().numpy()
                    im = ax.imshow(amap, cmap="viridis", vmin=0.0, vmax=1.0)
                    ax.set_title(
                        f"{stage_key.upper()} Map ({amap.shape[0]}x{amap.shape[1]})",
                        fontsize=8.5,
                    )
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.axis("off")

    fig.suptitle(
        "Spatial Attention Maps M_s(F) in (0, 1) (Feature Reweighting, Not Causal Proof)",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison(
    baseline_metrics: Dict[str, Any],
    attention_metrics: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Plot a side-by-side comparison chart of BaselineCNN vs. AttentionCNN."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    labels = ["Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    base_vals = [float(baseline_metrics.get(k, 0.0)) * 100.0 for k in metric_keys]
    attn_vals = [float(attention_metrics.get(k, 0.0)) * 100.0 for k in metric_keys]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    b1 = ax.bar(x - width / 2, base_vals, width, label="BaselineCNN", color="#1f4e78")
    b2 = ax.bar(x + width / 2, attn_vals, width, label="AttentionCNN", color="#d95f02")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("BaselineCNN vs. AttentionCNN — Evaluation Metrics (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(105.0, max(base_vals + attn_vals) + 12.0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    cfg_override: Optional[ExperimentConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Evaluate a saved checkpoint on the test split and generate metrics JSON, CSV, and plots."""
    raw_ckpt = load_checkpoint(checkpoint_path, device="cpu")
    cfg = cfg_override if cfg_override is not None else config_from_dict(raw_ckpt["config"])
    device = resolve_device(cfg.device)

    model = build_model(cfg.model).to(device)
    model.load_state_dict(raw_ckpt["model_state_dict"])
    model.eval()

    bundle = build_dataloaders(cfg.data, seed=cfg.seed)
    run_id = f"{cfg.model.name}_seed{cfg.seed}"
    res_dir = Path(output_dir) if output_dir is not None else (Path(cfg.training.results_dir) / run_id)
    res_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, test_loss = collect_predictions(model, bundle.test_loader, device)
    cls_metrics = compute_classification_metrics(y_true, y_pred, class_names=bundle.class_names)
    latency_info = measure_inference_latency(
        model=model,
        device=device,
        batch_size=cfg.evaluation.latency_batch_size,
        in_channels=cfg.model.in_channels,
        image_size=32,
        warmup_runs=cfg.evaluation.latency_warmup_runs,
        measure_runs=cfg.evaluation.latency_measure_runs,
    )

    param_count = count_parameters(model)
    history = raw_ckpt.get("history", [])
    plot_training_curves(history, res_dir, model_title=cfg.model.name)
    plot_confusion_matrix(
        cls_metrics["confusion_matrix"],
        bundle.class_names,
        res_dir / "confusion_matrix.png",
        title=f"{cfg.model.name} — Test Confusion Matrix",
    )
    plot_per_class_metrics(
        cls_metrics["per_class"],
        res_dir / "per_class_metrics.png",
        title=f"{cfg.model.name} — Per-Class Precision, Recall, and F1",
    )
    if isinstance(model, AttentionCNN):
        plot_attention_maps(
            model=model,
            dataset=bundle.test_dataset,
            device=device,
            output_path=res_dir / "attention_maps.png",
            mean=cfg.data.mean,
            std=cfg.data.std,
        )

    # Save per-class CSV
    per_class_csv = res_dir / "per_class_metrics.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["class_id", "class_name", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        writer.writerows(cls_metrics["per_class"])

    evaluation_report = {
        "run_id": run_id,
        "experiment_name": cfg.experiment_name,
        "run_type": cfg.run_type,
        "model_name": cfg.model.name,
        "seed": cfg.seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(raw_ckpt.get("epoch", 0)),
        "best_val_acc": float(raw_ckpt.get("best_val_acc", 0.0)),
        "is_synthetic_data": bundle.is_synthetic,
        "parameter_count": param_count,
        "test_loss": test_loss,
        "accuracy": cls_metrics["accuracy"],
        "macro_precision": cls_metrics["macro_precision"],
        "macro_recall": cls_metrics["macro_recall"],
        "macro_f1": cls_metrics["macro_f1"],
        "per_class": cls_metrics["per_class"],
        "confusion_matrix": cls_metrics["confusion_matrix"],
        "latency": latency_info,
    }
    save_json(evaluation_report, res_dir / "evaluation_metrics.json")
    return evaluation_report
