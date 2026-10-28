"""Structured YAML and CLI configuration management with strict validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

VALID_MODELS = ("baseline_cnn", "attention_cnn")
VALID_RUN_TYPES = ("smoke", "debug", "exploratory", "final_evaluation", "multi_seed")
VALID_SCHEDULERS = ("cosine", "step", "none")


@dataclass
class DataConfig:
    """Dataset splitting, normalization, and DataLoader parameters."""

    data_dir: str = "data"
    dataset: str = "cifar10"
    download: bool = True
    use_synthetic: bool = False
    synthetic_train_size: int = 240
    synthetic_test_size: int = 60
    val_fraction: float = 0.10
    split_seed: int = 42
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 2
    pin_memory: bool = True
    mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: List[float] = field(default_factory=lambda: [0.2470, 0.2435, 0.2616])
    random_crop_padding: int = 4
    horizontal_flip_prob: float = 0.5


@dataclass
class ModelConfig:
    """Architecture settings for BaselineCNN and AttentionCNN."""

    name: str = "baseline_cnn"
    num_classes: int = 10
    in_channels: int = 3
    channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    dropout_rate: float = 0.25
    attention_kernel_size: int = 7
    attention_residual: bool = True
    attention_stages: List[int] = field(default_factory=lambda: [2, 3])


@dataclass
class TrainingConfig:
    """Optimization, scheduling, and checkpoint settings."""

    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    step_size: int = 10
    gamma: float = 0.5
    min_lr: float = 0.00001
    grad_clip_norm: float = 2.0
    early_stopping_patience: int = 8
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    save_every_epoch: bool = False


@dataclass
class EvaluationConfig:
    """Inference latency measurement protocol settings."""

    latency_batch_size: int = 1
    latency_warmup_runs: int = 10
    latency_measure_runs: int = 50


@dataclass
class RobustnessConfig:
    """Controlled synthetic corruption evaluation settings."""

    corruption_seed: int = 1337
    gaussian_noise_sigmas: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    )
    gaussian_blur_sigmas: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0]
    )
    blur_kernel_size: int = 5


@dataclass
class ExperimentConfig:
    """Top-level configuration container for an AttentionVision experiment."""

    experiment_name: str = "cifar10_controlled_comparison"
    run_type: str = "final_evaluation"
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    def validate(self) -> None:
        """Validate configuration parameters and raise ValueError on invalid values."""
        if self.run_type not in VALID_RUN_TYPES:
            raise ValueError(
                f"Invalid run_type '{self.run_type}'. Must be one of {VALID_RUN_TYPES}."
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}.")
        if self.model.name not in VALID_MODELS:
            raise ValueError(
                f"Invalid model.name '{self.model.name}'. Must be one of {VALID_MODELS}."
            )
        if self.model.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.model.num_classes}.")
        if len(self.model.channels) != 3 or any(c <= 0 for c in self.model.channels):
            raise ValueError(
                f"model.channels must be a list of 3 positive integers, got {self.model.channels}."
            )
        if not (0.0 <= self.model.dropout_rate < 1.0):
            raise ValueError(
                f"model.dropout_rate must be in [0.0, 1.0), got {self.model.dropout_rate}."
            )
        if (
            self.model.attention_kernel_size <= 0
            or self.model.attention_kernel_size % 2 == 0
        ):
            raise ValueError(
                "model.attention_kernel_size must be a positive odd integer "
                f"(e.g., 3, 5, 7), got {self.model.attention_kernel_size}."
            )
        if any(s not in (1, 2, 3) for s in self.model.attention_stages):
            raise ValueError(
                f"model.attention_stages entries must be in {{1, 2, 3}}, got {self.model.attention_stages}."
            )
        if not (0.0 < self.data.val_fraction < 1.0):
            raise ValueError(
                f"data.val_fraction must be strictly between 0 and 1, got {self.data.val_fraction}."
            )
        if self.data.batch_size <= 0 or self.data.eval_batch_size <= 0:
            raise ValueError("Batch sizes must be strictly positive integers.")
        if self.training.epochs <= 0:
            raise ValueError(f"training.epochs must be > 0, got {self.training.epochs}.")
        if self.training.learning_rate <= 0.0:
            raise ValueError(
                f"training.learning_rate must be > 0, got {self.training.learning_rate}."
            )
        if self.training.weight_decay < 0.0:
            raise ValueError(
                f"training.weight_decay must be >= 0, got {self.training.weight_decay}."
            )
        if self.training.scheduler not in VALID_SCHEDULERS:
            raise ValueError(
                f"Invalid training.scheduler '{self.training.scheduler}'. "
                f"Must be one of {VALID_SCHEDULERS}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration dataclass hierarchy into a plain dictionary."""
        return asdict(self)

    def save_yaml(self, path: Union[str, Path]) -> Path:
        """Write configuration to a YAML file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(self.to_dict(), fp, sort_keys=False)
        return out_path


def config_from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    """Construct and validate an ExperimentConfig from a nested dictionary."""
    data_cfg = DataConfig(**raw.get("data", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))
    train_cfg = TrainingConfig(**raw.get("training", {}))
    eval_cfg = EvaluationConfig(**raw.get("evaluation", {}))
    rob_cfg = RobustnessConfig(**raw.get("robustness", {}))

    top_keys = {
        k: v
        for k, v in raw.items()
        if k in ("experiment_name", "run_type", "seed", "deterministic", "device")
    }
    cfg = ExperimentConfig(
        **top_keys,
        data=data_cfg,
        model=model_cfg,
        training=train_cfg,
        evaluation=eval_cfg,
        robustness=rob_cfg,
    )
    cfg.validate()
    return cfg


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file and apply optional overrides."""
    raw: Dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: '{path}'")
        with open(path, "r", encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp)
            if isinstance(loaded, dict):
                raw = loaded

    cfg = config_from_dict(raw)

    if overrides:
        if "experiment_name" in overrides and overrides["experiment_name"] is not None:
            cfg.experiment_name = str(overrides["experiment_name"])
        if "run_type" in overrides and overrides["run_type"] is not None:
            cfg.run_type = str(overrides["run_type"])
        if "seed" in overrides and overrides["seed"] is not None:
            cfg.seed = int(overrides["seed"])
        if "device" in overrides and overrides["device"] is not None:
            cfg.device = str(overrides["device"])
        if "model" in overrides and overrides["model"] is not None:
            cfg.model.name = str(overrides["model"])
        if "epochs" in overrides and overrides["epochs"] is not None:
            cfg.training.epochs = int(overrides["epochs"])
        if "batch_size" in overrides and overrides["batch_size"] is not None:
            cfg.data.batch_size = int(overrides["batch_size"])
        if "learning_rate" in overrides and overrides["learning_rate"] is not None:
            cfg.training.learning_rate = float(overrides["learning_rate"])
        if "weight_decay" in overrides and overrides["weight_decay"] is not None:
            cfg.training.weight_decay = float(overrides["weight_decay"])
        if "use_synthetic" in overrides and overrides["use_synthetic"] is not None:
            cfg.data.use_synthetic = bool(overrides["use_synthetic"])
        if "data_dir" in overrides and overrides["data_dir"] is not None:
            cfg.data.data_dir = str(overrides["data_dir"])
        if "checkpoint_dir" in overrides and overrides["checkpoint_dir"] is not None:
            cfg.training.checkpoint_dir = str(overrides["checkpoint_dir"])
        if "results_dir" in overrides and overrides["results_dir"] is not None:
            cfg.training.results_dir = str(overrides["results_dir"])
        cfg.validate()

    return cfg
