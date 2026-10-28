"""Data pipeline for CIFAR-10 and offline synthetic verification datasets.

Data Integrity Guarantees:
--------------------------
1. Disjoint Train/Validation Split:
   The 50,000 official CIFAR-10 training images are partitioned deterministically using
   `split_seed` into disjoint training (default 45,000) and validation (default 5,000) indices.
2. Transform Isolation (No Augmentation Leakage):
   Training and validation subsets wrap two separate underlying dataset instances:
   - `train_base` uses `build_transforms(data_cfg, is_train=True)` (random crop + horizontal flip).
   - `val_base` uses `build_transforms(data_cfg, is_train=False)` (deterministic normalization only).
   Thus, random training augmentations never affect validation or test evaluation.
3. Untouched Official Test Split:
   The 10,000 official CIFAR-10 test images are loaded exclusively for final evaluation and
   robustness testing, never for hyperparameter tuning or checkpoint selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from attentionvision.config import DataConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class SyntheticCIFAR10Dataset(Dataset):
    """Deterministic procedural 32x32 RGB dataset with localized class-specific spatial shapes.

    Used for offline unit tests, CI pipelines, and CPU smoke tests when external CIFAR-10
    downloads are disabled or unavailable. Each of the 10 classes renders a distinct colored
    geometric pattern at a randomized spatial location over low-amplitude background noise.
    """

    def __init__(
        self,
        num_samples: int = 240,
        num_classes: int = 10,
        seed: int = 42,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.transform = transform
        self.classes = list(CIFAR10_CLASSES[: self.num_classes])

        rng = np.random.default_rng(self.seed)
        self.data = np.zeros((self.num_samples, 32, 32, 3), dtype=np.uint8)
        self.targets: List[int] = []

        # Distinct RGB prototypes for each class
        palette = np.array(
            [
                [220, 60, 60],
                [60, 180, 75],
                [255, 225, 25],
                [0, 130, 200],
                [245, 130, 48],
                [145, 30, 180],
                [70, 240, 240],
                [240, 50, 230],
                [210, 245, 60],
                [250, 190, 212],
            ],
            dtype=np.float32,
        )

        for idx in range(self.num_samples):
            label = idx % self.num_classes
            self.targets.append(label)

            # Background clutter
            img = rng.uniform(25, 95, size=(32, 32, 3)).astype(np.float32)
            cx = int(rng.integers(8, 24))
            cy = int(rng.integers(8, 24))
            color = palette[label % len(palette)]

            # Draw class-specific localized spatial motif (8x8 region)
            yy, xx = np.ogrid[:32, :32]
            if label % 3 == 0:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 20
            elif label % 3 == 1:
                mask = (np.abs(xx - cx) <= 4) & (np.abs(yy - cy) <= 2)
            else:
                mask = (np.abs(xx - cx) + np.abs(yy - cy)) <= 5

            img[mask] = 0.80 * color + 0.20 * img[mask]
            # Add class-specific border stripe or corner marker
            stripe_row = (label * 3) % 30
            img[stripe_row : stripe_row + 2, 10:22, :] = color * 0.9

            self.data[idx] = np.clip(img, 0, 255).astype(np.uint8)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_arr = self.data[index]
        target = int(self.targets[index])
        img: Any = Image.fromarray(img_arr)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def build_transforms(data_cfg: DataConfig, is_train: bool = True) -> transforms.Compose:
    """Create training or evaluation image transformations.

    Args:
        data_cfg: Data configuration containing normalization stats and augmentation params.
        is_train: If True, includes RandomCrop and RandomHorizontalFlip before normalization.
            If False, applies only deterministic ToTensor and Normalize.
    """
    ops: List[Callable[[Any], Any]] = []
    if is_train:
        if data_cfg.random_crop_padding > 0:
            ops.append(transforms.RandomCrop(32, padding=data_cfg.random_crop_padding))
        if data_cfg.horizontal_flip_prob > 0.0:
            ops.append(transforms.RandomHorizontalFlip(p=data_cfg.horizontal_flip_prob))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg.mean, std=data_cfg.std),
        ]
    )
    return transforms.Compose(ops)


def compute_train_val_indices(
    total_size: int,
    val_fraction: float = 0.10,
    split_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Deterministically partition [0, total_size) into disjoint train and validation indices."""
    if total_size < 2:
        raise ValueError(f"total_size must be >= 2, got {total_size}.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")

    val_size = max(1, int(round(total_size * val_fraction)))
    train_size = total_size - val_size
    if train_size < 1:
        raise ValueError("Split resulted in 0 training samples; reduce val_fraction.")

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(total_size).tolist()
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]
    return train_indices, val_indices


@dataclass
class DataBundle:
    """Container holding train, validation, and test datasets, indices, and DataLoaders."""

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_indices: List[int]
    val_indices: List[int]
    class_names: Tuple[str, ...]
    is_synthetic: bool


def build_dataloaders(data_cfg: DataConfig, seed: int = 42) -> DataBundle:
    """Construct train, validation, and test DataLoaders with strict split isolation."""
    train_tf = build_transforms(data_cfg, is_train=True)
    eval_tf = build_transforms(data_cfg, is_train=False)

    if data_cfg.use_synthetic:
        train_base: Dataset = SyntheticCIFAR10Dataset(
            num_samples=data_cfg.synthetic_train_size,
            num_classes=10,
            seed=data_cfg.split_seed,
            transform=train_tf,
        )
        val_base: Dataset = SyntheticCIFAR10Dataset(
            num_samples=data_cfg.synthetic_train_size,
            num_classes=10,
            seed=data_cfg.split_seed,
            transform=eval_tf,
        )
        test_dataset: Dataset = SyntheticCIFAR10Dataset(
            num_samples=data_cfg.synthetic_test_size,
            num_classes=10,
            seed=data_cfg.split_seed + 999,
            transform=eval_tf,
        )
        is_synthetic = True
    else:
        root_path = Path(data_cfg.data_dir)
        try:
            train_base = datasets.CIFAR10(
                root=str(root_path),
                train=True,
                transform=train_tf,
                download=data_cfg.download,
            )
            val_base = datasets.CIFAR10(
                root=str(root_path),
                train=True,
                transform=eval_tf,
                download=False,
            )
            test_dataset = datasets.CIFAR10(
                root=str(root_path),
                train=False,
                transform=eval_tf,
                download=False,
            )
            is_synthetic = False
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load or download CIFAR-10 in '{root_path}'. "
                "Verify internet connectivity or pass '--synthetic' (use_synthetic: true) "
                f"for offline testing. Underlying error: {exc}"
            ) from exc

    train_indices, val_indices = compute_train_val_indices(
        total_size=len(train_base),
        val_fraction=data_cfg.val_fraction,
        split_seed=data_cfg.split_seed,
    )

    train_subset = Subset(train_base, train_indices)
    val_subset = Subset(val_base, val_indices)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(seed))

    use_pin = bool(data_cfg.pin_memory and torch.cuda.is_available())

    train_loader = DataLoader(
        train_subset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=use_pin,
        generator=loader_gen,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=data_cfg.eval_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.eval_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=use_pin,
    )

    return DataBundle(
        train_dataset=train_subset,
        val_dataset=val_subset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_indices=train_indices,
        val_indices=val_indices,
        class_names=CIFAR10_CLASSES,
        is_synthetic=is_synthetic,
    )


def inspect_dataset_bundle(bundle: DataBundle) -> Dict[str, Any]:
    """Return a summary dictionary verifying shapes, value ranges, and split integrity."""
    sample_img, sample_label = bundle.train_dataset[0]
    overlap = set(bundle.train_indices).intersection(set(bundle.val_indices))
    return {
        "is_synthetic": bundle.is_synthetic,
        "train_size": len(bundle.train_dataset),
        "val_size": len(bundle.val_dataset),
        "test_size": len(bundle.test_dataset),
        "disjoint_train_val": len(overlap) == 0,
        "sample_shape": list(sample_img.shape),
        "sample_dtype": str(sample_img.dtype),
        "sample_min": round(float(sample_img.min().item()), 4),
        "sample_max": round(float(sample_img.max().item()), 4),
        "sample_label": int(sample_label),
        "num_classes": len(bundle.class_names),
        "class_names": list(bundle.class_names),
    }


def denormalize_image(
    img_tensor: torch.Tensor,
    mean: Sequence[float] = (0.4914, 0.4822, 0.4465),
    std: Sequence[float] = (0.2470, 0.2435, 0.2616),
) -> np.ndarray:
    """Convert a normalized (C, H, W) tensor back to a displayable (H, W, C) float array in [0, 1]."""
    arr = img_tensor.detach().cpu().numpy()
    m = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    unnorm = np.clip(arr * s + m, 0.0, 1.0)
    return np.transpose(unnorm, (1, 2, 0))


def save_sample_grid(
    dataset: Dataset,
    output_path: Union[str, Path],
    mean: Sequence[float] = (0.4914, 0.4822, 0.4465),
    std: Sequence[float] = (0.2470, 0.2435, 0.2616),
    num_samples: int = 10,
) -> Path:
    """Save a grid of sample images and class labels to a PNG file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = min(num_samples, len(dataset))
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx < n:
            img_t, label = dataset[idx]
            vis = denormalize_image(img_t, mean=mean, std=std)
            ax.imshow(vis)
            cls_name = CIFAR10_CLASSES[int(label)] if int(label) < len(CIFAR10_CLASSES) else str(label)
            ax.set_title(f"{cls_name} ({int(label)})", fontsize=9)
        ax.axis("off")

    fig.suptitle("Dataset Sample Inspection Grid (32x32 RGB)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
