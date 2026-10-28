"""Unit tests for dataset splitting, transform isolation, and synthetic data pipeline."""

from __future__ import annotations

import torch

from attentionvision.config import DataConfig
from attentionvision.data import (
    SyntheticCIFAR10Dataset,
    build_dataloaders,
    compute_train_val_indices,
    inspect_dataset_bundle,
)


def test_deterministic_and_disjoint_train_val_split() -> None:
    """Verify that train/val split indices are strictly disjoint and reproducible."""
    train_a, val_a = compute_train_val_indices(total_size=500, val_fraction=0.20, split_seed=42)
    train_b, val_b = compute_train_val_indices(total_size=500, val_fraction=0.20, split_seed=42)
    train_c, val_c = compute_train_val_indices(total_size=500, val_fraction=0.20, split_seed=99)

    assert len(train_a) == 400
    assert len(val_a) == 100
    assert set(train_a).isdisjoint(set(val_a))
    assert set(train_a) | set(val_a) == set(range(500))
    assert train_a == train_b and val_a == val_b
    assert train_a != train_c


def test_synthetic_dataloaders_shapes_and_label_ranges() -> None:
    """Verify that offline synthetic DataLoaders yield valid (B, 3, 32, 32) tensors and labels in [0, 9]."""
    cfg = DataConfig(
        use_synthetic=True,
        synthetic_train_size=100,
        synthetic_test_size=30,
        val_fraction=0.20,
        batch_size=16,
        eval_batch_size=15,
        num_workers=0,
    )
    bundle = build_dataloaders(cfg, seed=42)
    info = inspect_dataset_bundle(bundle)

    assert info["disjoint_train_val"] is True
    assert info["train_size"] == 80
    assert info["val_size"] == 20
    assert info["test_size"] == 30

    images, labels = next(iter(bundle.train_loader))
    assert images.shape == (16, 3, 32, 32)
    assert labels.shape == (16,)
    assert int(labels.min().item()) >= 0
    assert int(labels.max().item()) < 10


def test_validation_transform_is_deterministic_without_augmentation_leakage() -> None:
    """Verify that validation dataset samples do not undergo stochastic training augmentations."""
    cfg = DataConfig(
        use_synthetic=True,
        synthetic_train_size=60,
        synthetic_test_size=20,
        val_fraction=0.25,
        random_crop_padding=4,
        horizontal_flip_prob=0.5,
    )
    bundle = build_dataloaders(cfg, seed=42)

    # Fetching the same validation sample twice must yield identical tensors
    img1, lbl1 = bundle.val_dataset[0]
    img2, lbl2 = bundle.val_dataset[0]
    assert lbl1 == lbl2
    assert torch.allclose(img1, img2, atol=1e-7)


def test_synthetic_dataset_direct_instantiation() -> None:
    """Verify SyntheticCIFAR10Dataset produces 32x32 RGB PIL images when transform=None."""
    ds = SyntheticCIFAR10Dataset(num_samples=20, num_classes=10, seed=7, transform=None)
    assert len(ds) == 20
    img, label = ds[0]
    assert img.size == (32, 32)
    assert 0 <= label < 10
