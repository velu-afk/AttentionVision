"""Standalone script to compare completed AttentionVision experiments and print a Markdown table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from attentionvision.experiment import compare_experiments


def format_markdown_table(aggregated: dict) -> str:
    """Format aggregated model statistics as a readable Markdown table."""
    if not aggregated:
        return "No completed evaluation runs found in the specified results directory."

    headers = [
        "Model",
        "Runs (Seeds)",
        "Data Type",
        "Params",
        "Val Acc (%)",
        "Test Acc (%)",
        "Macro-F1 (%)",
        "Train Time (s)",
        "Latency (ms/img)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for model_name, stats in sorted(aggregated.items()):
        n_runs = stats["num_runs"]
        seeds_str = ",".join(str(s) for s in stats["seeds"])
        data_type = "Synthetic (Smoke)" if stats["is_synthetic_data"] else "Official CIFAR-10"
        params = f"{stats['parameter_count']:,}"

        val_acc = f"{stats['best_val_acc_mean'] * 100:.2f} +/- {stats['best_val_acc_std'] * 100:.2f}"
        test_acc = f"{stats['accuracy_mean'] * 100:.2f} +/- {stats['accuracy_std'] * 100:.2f}"
        macro_f1 = f"{stats['macro_f1_mean'] * 100:.2f} +/- {stats['macro_f1_std'] * 100:.2f}"

        if stats.get("training_time_sec_mean") is not None:
            t_str = f"{stats['training_time_sec_mean']:.2f} +/- {stats['training_time_sec_std']:.2f}"
        else:
            t_str = "N/A"

        lat_str = f"{stats['latency_mean_ms_mean']:.3f}"
        row = [
            f"`{model_name}`",
            f"{n_runs} ({seeds_str})",
            data_type,
            params,
            val_acc,
            test_acc,
            macro_f1,
            t_str,
            lat_str,
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare completed AttentionVision runs and generate summary CSV, JSON, and Markdown tables."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment run folders (default: results).",
    )
    args = parser.parse_args(argv)

    res_dir = Path(args.results_dir)
    summary = compare_experiments(res_dir)
    md_table = format_markdown_table(summary["aggregated_by_model"])
    print("\n=== Experiment Comparison Summary ===")
    print(md_table)

    md_path = res_dir / "comparison_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_table + "\n", encoding="utf-8")
    print(f"\nSaved Markdown table to: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
