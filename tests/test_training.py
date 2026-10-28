"""Unit tests for the training loop, resume-from-checkpoint workflow, and config validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from attentionvision.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from attentionvision.evaluate import evaluate_checkpoint
from attentionvision.models import AttentionCNN, BaselineCNN
from attentionvision.train import (
    build_optimizer_and_scheduler,
    run_training,
    train_one_epoch,
    validate_epoch,
)


@pytest.mark.parametrize("model_cls", [BaselineCNN, AttentionCNN])
def test_single_epoch_train_and_validate_on_synthetic_tensors(model_cls: type) -> None:
    """Verify minimal optimization step, finite loss, parameter update, and validation loop."""
    torch.manual_seed(42)
    model = model_cls()
    initial_w = model.classifier.weight.clone().detach()

    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, size=(16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)

    cfg = ExperimentConfig()
    optimizer, _ = build_optimizer_and_scheduler(model, cfg)
    criterion = torch.nn.CrossEntropyLoss()

    train_stats = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    assert train_stats["loss"] > 0.0
    assert 0.0 <= train_stats["accuracy"] <= 1.0
    # Verify parameters updated after optimizer step
    assert not torch.allclose(initial_w, model.classifier.weight.detach())

    val_stats = validate_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )
    assert val_stats["loss"] > 0.0
    assert 0.0 <= val_stats["accuracy"] <= 1.0


def test_end_to_end_run_training_and_resume(tmp_path: Path) -> None:
    """Verify full training pipeline, checkpoint creation, resume behavior, and evaluation."""
    cfg = ExperimentConfig(
        experiment_name="unit_test_run",
        run_type="smoke",
        seed=42,
        device="cpu",
        data=DataConfig(
            use_synthetic=True,
            synthetic_train_size=40,
            synthetic_test_size=16,
            val_fraction=0.25,
            batch_size=10,
            eval_batch_size=8,
            num_workers=0,
        ),
        model=ModelConfig(name="attention_cnn"),
        training=TrainingConfig(
            epochs=1,
            learning_rate=0.002,
            checkpoint_dir=str(tmp_path / "ckpts"),
            results_dir=str(tmp_path / "results"),
        ),
    )

    summary_ep1 = run_training(cfg)
    assert summary_ep1["epochs_completed"] == 1
    assert Path(summary_ep1["best_checkpoint"]).exists()

    # Resume training up to epoch 2
    cfg.training.epochs = 2
    summary_ep2 = run_training(cfg, resume_from=summary_ep1["last_checkpoint"])
    assert summary_ep2["epochs_completed"] == 2

    # Evaluate saved checkpoint
    eval_report = evaluate_checkpoint(summary_ep2["best_checkpoint"], cfg_override=cfg)
    assert 0.0 <= eval_report["accuracy"] <= 1.0
    assert Path(cfg.training.results_dir, "attention_cnn_seed42", "evaluation_metrics.json").exists()


def test_invalid_configuration_raises_value_error() -> None:
    """Verify that invalid model names, learning rates, or validation fractions fail fast."""
    cfg = ExperimentConfig()
    cfg.model.name = "invalid_transformer"
    with pytest.raises(ValueError, match="Invalid model.name"):
        cfg.validate()

    cfg2 = ExperimentConfig()
    cfg2.training.learning_rate = -0.01
    with pytest.raises(ValueError, match="training.learning_rate"):
        cfg2.validate()

    cfg3 = ExperimentConfig()
    cfg3.data.val_fraction = 1.2
    with pytest.raises(ValueError, match="data.val_fraction"):
        cfg3.validate()
