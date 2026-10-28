"""Readable, beginner-friendly training loop, validation loop, and checkpoint management."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attentionvision.config import ExperimentConfig, config_from_dict
from attentionvision.data import build_dataloaders
from attentionvision.models import build_model
from attentionvision.utils import (
    count_parameters,
    get_environment_info,
    resolve_device,
    save_json,
    set_seed,
    setup_logger,
)


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: ExperimentConfig,
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    """Create the AdamW optimizer and optional learning-rate scheduler."""
    opt_name = cfg.training.optimizer.strip().lower()
    if opt_name == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=0.9,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{cfg.training.optimizer}'.")

    sched_name = cfg.training.scheduler.strip().lower()
    scheduler: Optional[Any] = None
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.training.min_lr,
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.step_size,
            gamma=cfg.training.gamma,
        )
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 2.0,
) -> Dict[str, float]:
    """Run one complete training epoch over `loader`.

    Beginner Walkthrough:
        1. `model.train()` enables training-specific layers (Dropout, BatchNorm batch stats).
        2. For each mini-batch `(images, labels)`:
           - Move tensors to `device`.
           - Clear previous gradients with `optimizer.zero_grad()`.
           - Compute class logits via the forward pass `logits = model(images)`.
           - Compute cross-entropy loss `loss = criterion(logits, labels)`.
           - Compute gradients via backpropagation `loss.backward()`.
           - Optionally clip gradient norm for numerical stability.
           - Update parameters with `optimizer.step()`.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError(f"Non-finite training loss encountered: {loss.item()}")

        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)
    return {"loss": round(avg_loss, 6), "accuracy": round(accuracy, 6)}


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate `model` on `loader` without gradient updates.

    Beginner Walkthrough:
        1. `model.eval()` switches BatchNorm to use running statistics and disables Dropout.
        2. `with torch.no_grad():` disables gradient graph construction to save memory and
           guarantee that validation data never influences model weights.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)
    return {"loss": round(avg_loss, 6), "accuracy": round(accuracy, 6)}


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_val_acc: float,
    best_val_loss: float,
    cfg: ExperimentConfig,
    history: List[Dict[str, Any]],
) -> Path:
    """Save a complete, self-contained training checkpoint for evaluation or resumption."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "seed": int(cfg.seed),
        "model_name": cfg.model.name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": cfg.to_dict(),
        "history": history,
        "environment": get_environment_info(),
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Safely load a saved checkpoint and optionally restore model/optimizer/scheduler states."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: '{ckpt_path}'")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    required_keys = {"epoch", "model_state_dict", "config"}
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        raise ValueError(
            f"Invalid checkpoint structure in '{ckpt_path}'. Missing required keys."
        )

    if model is not None:
        model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return payload


def run_training(
    cfg: ExperimentConfig,
    resume_from: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute the full training and validation workflow for a configured model."""
    cfg.validate()
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = resolve_device(cfg.device)

    run_id = f"{cfg.model.name}_seed{cfg.seed}"
    ckpt_dir = Path(cfg.training.checkpoint_dir) / run_id
    res_dir = Path(cfg.training.results_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(f"attentionvision.{run_id}", log_file=res_dir / "train.log")
    logger.info(
        f"Starting run '{run_id}' | run_type={cfg.run_type} | device={device} | seed={cfg.seed}"
    )

    bundle = build_dataloaders(cfg.data, seed=cfg.seed)
    model = build_model(cfg.model).to(device)
    param_count = count_parameters(model)
    logger.info(
        f"Model={cfg.model.name} | parameters={param_count:,} | "
        f"train={len(bundle.train_dataset)} | val={len(bundle.val_dataset)} | "
        f"synthetic={bundle.is_synthetic}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    start_epoch = 1
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    if resume_from is not None:
        ckpt_payload = load_checkpoint(
            resume_from,
            device=device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(ckpt_payload["epoch"]) + 1
        best_val_acc = float(ckpt_payload.get("best_val_acc", -1.0))
        best_val_loss = float(ckpt_payload.get("best_val_loss", float("inf")))
        history = list(ckpt_payload.get("history", []))
        logger.info(f"Resumed from '{resume_from}' at epoch {start_epoch}.")

    best_ckpt_path = ckpt_dir / "best_model.pt"
    last_ckpt_path = ckpt_dir / "last_model.pt"
    epochs_without_improvement = 0
    t_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        ep_t0 = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_metrics = train_one_epoch(
            model=model,
            loader=bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=cfg.training.grad_clip_norm,
        )
        val_metrics = validate_epoch(
            model=model,
            loader=bundle.val_loader,
            criterion=criterion,
            device=device,
        )
        if scheduler is not None:
            scheduler.step()

        ep_duration = time.perf_counter() - ep_t0
        record = {
            "epoch": epoch,
            "lr": round(current_lr, 8),
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "epoch_time_sec": round(ep_duration, 3),
        }
        history.append(record)

        improved = (val_metrics["accuracy"] > best_val_acc) or (
            val_metrics["accuracy"] == best_val_acc and val_metrics["loss"] < best_val_loss
        )
        if improved:
            best_val_acc = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc,
                best_val_loss=best_val_loss,
                cfg=cfg,
                history=history,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_acc=best_val_acc,
            best_val_loss=best_val_loss,
            cfg=cfg,
            history=history,
        )

        logger.info(
            f"Epoch {epoch:02d}/{cfg.training.epochs:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
            f"time={ep_duration:.2f}s"
        )

        if (
            cfg.training.early_stopping_patience > 0
            and epochs_without_improvement >= cfg.training.early_stopping_patience
        ):
            logger.info(
                f"Early stopping triggered after {epochs_without_improvement} epochs without "
                "validation improvement."
            )
            break

    total_time_sec = round(time.perf_counter() - t_start, 3)

    # Save configuration and CSV epoch history
    cfg.save_yaml(res_dir / "config.yaml")
    history_csv = res_dir / "history.csv"
    if history:
        with open(history_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    summary = {
        "run_id": run_id,
        "experiment_name": cfg.experiment_name,
        "run_type": cfg.run_type,
        "model_name": cfg.model.name,
        "seed": cfg.seed,
        "device": str(device),
        "is_synthetic_data": bundle.is_synthetic,
        "parameter_count": param_count,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 6),
        "best_val_loss": round(best_val_loss, 6),
        "total_training_time_sec": total_time_sec,
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "results_dir": str(res_dir),
        "environment": get_environment_info(),
    }
    save_json(summary, res_dir / "training_summary.json")
    return summary
