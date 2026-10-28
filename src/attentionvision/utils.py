"""Utility functions for reproducibility, device selection, logging, and environment inspection."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def setup_logger(
    name: str = "attentionvision",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a console (and optional file) logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducible experiments.

    Note on reproducibility:
        Setting seeds guarantees identical weight initialization and deterministic
        train/validation splits on the same hardware and software stack. Cross-device
        (CPU vs. GPU) bitwise identity is not guaranteed due to floating-point atomic
        addition order in parallel kernels.
    """
    if seed < 0:
        raise ValueError(f"Random seed must be non-negative, got {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the execution device safely, falling back to CPU when CUDA is unavailable."""
    req = requested.strip().lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(req)
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device specification '{requested}'. Use 'auto', 'cpu', or 'cuda'.")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Return the total number of parameters in a PyTorch module."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_ram_gb() -> Optional[float]:
    """Read total system memory in GiB on Linux where /proc/meminfo is available."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024.0 * 1024.0), 2)
        except Exception:
            return None
    return None


def get_environment_info() -> Dict[str, Any]:
    """Collect hardware, operating system, and package version metadata for experiment logs."""
    import matplotlib
    import pandas
    import scipy
    import sklearn
    import torchvision
    import yaml

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = (
        torch.cuda.get_device_name(0)
        if cuda_available and torch.cuda.device_count() > 0
        else None
    )

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "ram_gb": get_ram_gb(),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_device_name": cuda_device_name,
        "packages": {
            "torch": getattr(torch, "__version__", "unknown"),
            "torchvision": getattr(torchvision, "__version__", "unknown"),
            "numpy": getattr(np, "__version__", "unknown"),
            "scipy": getattr(scipy, "__version__", "unknown"),
            "pandas": getattr(pandas, "__version__", "unknown"),
            "matplotlib": getattr(matplotlib, "__version__", "unknown"),
            "scikit_learn": getattr(sklearn, "__version__", "unknown"),
            "pyyaml": getattr(yaml, "__version__", "unknown"),
        },
    }


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Serialize a dictionary to a human-readable JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=False)
    return out_path
