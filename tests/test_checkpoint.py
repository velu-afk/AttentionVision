"""Unit tests for checkpoint saving, loading, state restoration, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from attentionvision.config import ExperimentConfig
from attentionvision.models import AttentionCNN
from attentionvision.train import (
    build_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_save_and_restore_roundtrip(tmp_path: Path) -> None:
    """Verify that saving and loading a checkpoint restores exact model and optimizer states."""
    cfg = ExperimentConfig()
    cfg.model.name = "attention_cnn"

    model1 = AttentionCNN()
    opt1, sched1 = build_optimizer_and_scheduler(model1, cfg)

    # Perform one synthetic backward/optimizer step so optimizer state is populated
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    loss = torch.nn.CrossEntropyLoss()(model1(x), y)
    loss.backward()
    opt1.step()
    if sched1 is not None:
        sched1.step()

    ckpt_file = tmp_path / "test_ckpt.pt"
    history = [{"epoch": 1, "train_loss": 2.1, "val_loss": 2.0, "train_acc": 0.25, "val_acc": 0.30}]
    save_checkpoint(
        path=ckpt_file,
        model=model1,
        optimizer=opt1,
        scheduler=sched1,
        epoch=1,
        best_val_acc=0.30,
        best_val_loss=2.0,
        cfg=cfg,
        history=history,
    )

    assert ckpt_file.exists()

    model2 = AttentionCNN()
    opt2, sched2 = build_optimizer_and_scheduler(model2, cfg)
    loaded = load_checkpoint(
        path=ckpt_file,
        device="cpu",
        model=model2,
        optimizer=opt2,
        scheduler=sched2,
    )

    assert loaded["epoch"] == 1
    assert loaded["best_val_acc"] == pytest.approx(0.30)
    assert loaded["model_name"] == "attention_cnn"

    model1.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_missing_checkpoint_raises_file_not_found(tmp_path: Path) -> None:
    """Verify informative FileNotFoundError when loading a nonexistent checkpoint."""
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        _ = load_checkpoint(tmp_path / "nonexistent.pt")
