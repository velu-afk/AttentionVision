"""Command-Line Interface (CLI) for AttentionVision."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from attentionvision.config import load_config
from attentionvision.data import build_dataloaders, inspect_dataset_bundle, save_sample_grid
from attentionvision.evaluate import evaluate_checkpoint
from attentionvision.experiment import compare_experiments
from attentionvision.robustness import run_robustness_experiment
from attentionvision.train import run_training
from attentionvision.utils import get_environment_info


def _add_common_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline_cnn", "attention_cnn"],
        default=None,
        help="Override model architecture ('baseline_cnn' or 'attention_cnn').",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override execution device ('auto', 'cpu', or 'cuda').",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override optimizer learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None, help="Override optimizer weight decay."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic 32x32 RGB data (no internet download required).",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset directory.")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Override checkpoint output directory."
    )
    parser.add_argument(
        "--results-dir", type=str, default=None, help="Override results output directory."
    )


def _extract_overrides(args: argparse.Namespace) -> dict:
    overrides = {
        "model": getattr(args, "model", None),
        "seed": getattr(args, "seed", None),
        "device": getattr(args, "device", None),
        "epochs": getattr(args, "epochs", None),
        "batch_size": getattr(args, "batch_size", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "data_dir": getattr(args, "data_dir", None),
        "checkpoint_dir": getattr(args, "checkpoint_dir", None),
        "results_dir": getattr(args, "results_dir", None),
    }
    if getattr(args, "synthetic", False):
        overrides["use_synthetic"] = True
    return overrides


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="attentionvision",
        description=(
            "AttentionVision CLI: Controlled research framework investigating lightweight "
            "spatial attention in CNN-based CIFAR-10 image classification."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. inspect-env
    subparsers.add_parser(
        "inspect-env",
        help="Inspect Python, PyTorch, CPU/RAM, and CUDA availability.",
    )

    # 2. prepare-data
    prep_parser = subparsers.add_parser(
        "prepare-data",
        help="Prepare/verify dataset splits and save a sample image grid.",
    )
    _add_common_overrides(prep_parser)
    prep_parser.add_argument(
        "--sample-grid-out",
        type=str,
        default="results/sample_grid.png",
        help="Output path for sample visualization grid.",
    )

    # 3. train
    train_parser = subparsers.add_parser(
        "train",
        help="Train BaselineCNN or AttentionCNN and save checkpoints and logs.",
    )
    _add_common_overrides(train_parser)
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to a saved checkpoint (.pt) to resume training from.",
    )

    # 4. evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved checkpoint on the test set and generate plots.",
    )
    _add_common_overrides(eval_parser)
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt) to evaluate.",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for evaluation metrics and plots.",
    )

    # 5. robustness
    rob_parser = subparsers.add_parser(
        "robustness",
        help="Run Gaussian noise and Gaussian blur corruption experiments on both models.",
    )
    _add_common_overrides(rob_parser)
    rob_parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        required=True,
        help="Path to trained BaselineCNN checkpoint (.pt).",
    )
    rob_parser.add_argument(
        "--attention-checkpoint",
        type=str,
        required=True,
        help="Path to trained AttentionCNN checkpoint (.pt).",
    )
    rob_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for robustness CSV, JSON, and plots.",
    )

    # 6. smoke-test
    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Run a rapid end-to-end CPU smoke test (train + evaluate + robustness + compare).",
    )
    smoke_parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke.yaml",
        help="Smoke test YAML configuration (default: configs/smoke.yaml).",
    )

    # 7. compare
    cmp_parser = subparsers.add_parser(
        "compare",
        help="Compare completed experiment runs and generate summary CSV/JSON/plots.",
    )
    cmp_parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root results directory to scan for completed runs.",
    )

    # 8. plot
    plot_parser = subparsers.add_parser(
        "plot",
        help="Regenerate evaluation and comparison plots for a results directory.",
    )
    plot_parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory containing completed runs.",
    )

    # 9. test
    test_parser = subparsers.add_parser(
        "test",
        help="Execute the offline pytest unit test suite.",
    )
    test_parser.add_argument(
        "--test-path",
        type=str,
        default="tests",
        help="Path to test directory or test file.",
    )

    # 10. lint
    lint_parser = subparsers.add_parser(
        "lint",
        help="Run Ruff linter and format verification.",
    )
    lint_parser.add_argument(
        "--targets",
        nargs="*",
        default=["src", "tests", "scripts"],
        help="Directories or files to lint.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entrypoint for the `attentionvision` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.command

    if cmd == "inspect-env":
        info = get_environment_info()
        print(json.dumps(info, indent=2))
        return 0

    if cmd == "prepare-data":
        cfg = load_config(args.config, overrides=_extract_overrides(args))
        bundle = build_dataloaders(cfg.data, seed=cfg.seed)
        summary = inspect_dataset_bundle(bundle)
        grid_path = save_sample_grid(
            bundle.val_dataset,
            output_path=args.sample_grid_out,
            mean=cfg.data.mean,
            std=cfg.data.std,
        )
        summary["sample_grid_path"] = str(grid_path)
        print(json.dumps(summary, indent=2))
        return 0

    if cmd == "train":
        cfg = load_config(args.config, overrides=_extract_overrides(args))
        summary = run_training(cfg, resume_from=args.resume)
        print(json.dumps(summary, indent=2))
        return 0

    if cmd == "evaluate":
        cfg = load_config(args.config, overrides=_extract_overrides(args))
        report = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            cfg_override=cfg,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "run_id": report["run_id"],
                    "model_name": report["model_name"],
                    "accuracy": report["accuracy"],
                    "macro_f1": report["macro_f1"],
                    "parameter_count": report["parameter_count"],
                    "latency_mean_ms": report["latency"]["mean_sample_ms"],
                },
                indent=2,
            )
        )
        return 0

    if cmd == "robustness":
        cfg = load_config(args.config, overrides=_extract_overrides(args))
        rob_summary = run_robustness_experiment(
            cfg=cfg,
            baseline_checkpoint=args.baseline_checkpoint,
            attention_checkpoint=args.attention_checkpoint,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "csv_path": rob_summary["csv_path"],
                    "noise_plot": rob_summary["noise_plot"],
                    "blur_plot": rob_summary["blur_plot"],
                    "num_conditions": len(rob_summary["records"]),
                },
                indent=2,
            )
        )
        return 0

    if cmd == "smoke-test":
        base_cfg = load_config(args.config, overrides={"model": "baseline_cnn"})
        attn_cfg = load_config(args.config, overrides={"model": "attention_cnn"})
        base_train = run_training(base_cfg)
        attn_train = run_training(attn_cfg)
        base_eval = evaluate_checkpoint(base_train["best_checkpoint"], cfg_override=base_cfg)
        attn_eval = evaluate_checkpoint(attn_train["best_checkpoint"], cfg_override=attn_cfg)
        rob = run_robustness_experiment(
            cfg=base_cfg,
            baseline_checkpoint=base_train["best_checkpoint"],
            attention_checkpoint=attn_train["best_checkpoint"],
        )
        cmp_res = compare_experiments(base_cfg.training.results_dir)
        print(
            json.dumps(
                {
                    "status": "smoke_test_completed",
                    "baseline_test_acc": base_eval["accuracy"],
                    "attention_test_acc": attn_eval["accuracy"],
                    "robustness_csv": rob["csv_path"],
                    "total_runs_compared": cmp_res["total_runs_found"],
                },
                indent=2,
            )
        )
        return 0

    if cmd in ("compare", "plot"):
        cmp_res = compare_experiments(args.results_dir)
        print(
            json.dumps(
                {
                    "results_dir": cmp_res["results_dir"],
                    "total_runs_found": cmp_res["total_runs_found"],
                    "aggregated_by_model": cmp_res["aggregated_by_model"],
                },
                indent=2,
            )
        )
        return 0

    if cmd == "test":
        proc = subprocess.run([sys.executable, "-m", "pytest", args.test_path, "-v"], check=False)
        return proc.returncode

    if cmd == "lint":
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", *args.targets],
            check=False,
        )
        return proc.returncode

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
