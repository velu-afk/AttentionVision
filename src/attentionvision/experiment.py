"""Experiment aggregation, multi-seed statistical summary, and comparison table generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from attentionvision.evaluate import plot_model_comparison
from attentionvision.utils import save_json


def collect_completed_runs(results_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Scan `results_dir` for completed `evaluation_metrics.json` files and merge training stats."""
    root = Path(results_dir)
    if not root.exists():
        return []

    runs: List[Dict[str, Any]] = []
    for eval_file in sorted(root.rglob("evaluation_metrics.json")):
        try:
            eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        train_file = eval_file.parent / "training_summary.json"
        train_time = None
        if train_file.exists():
            try:
                train_data = json.loads(train_file.read_text(encoding="utf-8"))
                train_time = train_data.get("total_training_time_sec")
            except Exception:
                train_time = None

        latency = eval_data.get("latency", {})
        runs.append(
            {
                "run_id": eval_data.get("run_id", eval_file.parent.name),
                "experiment_name": eval_data.get("experiment_name", "unknown"),
                "run_type": eval_data.get("run_type", "unknown"),
                "model_name": eval_data.get("model_name", "unknown"),
                "seed": int(eval_data.get("seed", 42)),
                "is_synthetic_data": bool(eval_data.get("is_synthetic_data", False)),
                "parameter_count": int(eval_data.get("parameter_count", 0)),
                "best_val_acc": float(eval_data.get("best_val_acc", 0.0)),
                "test_loss": float(eval_data.get("test_loss", 0.0)),
                "accuracy": float(eval_data.get("accuracy", 0.0)),
                "macro_precision": float(eval_data.get("macro_precision", 0.0)),
                "macro_recall": float(eval_data.get("macro_recall", 0.0)),
                "macro_f1": float(eval_data.get("macro_f1", 0.0)),
                "training_time_sec": round(float(train_time), 3) if train_time is not None else None,
                "latency_mean_ms": float(latency.get("mean_sample_ms", 0.0)),
                "latency_std_ms": float(latency.get("std_batch_ms", 0.0)),
                "latency_device": str(latency.get("device", "cpu")),
                "latency_batch_size": int(latency.get("batch_size", 1)),
            }
        )
    return runs


def aggregate_by_model(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics by `model_name`, reporting mean and standard deviation across seeds."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        grouped.setdefault(str(r["model_name"]), []).append(r)

    summary: Dict[str, Dict[str, Any]] = {}
    metric_keys = [
        "best_val_acc",
        "test_loss",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "latency_mean_ms",
    ]
    for model_name, items in grouped.items():
        agg: Dict[str, Any] = {
            "model_name": model_name,
            "num_runs": len(items),
            "seeds": [int(i["seed"]) for i in items],
            "run_types": sorted(set(str(i["run_type"]) for i in items)),
            "is_synthetic_data": any(bool(i["is_synthetic_data"]) for i in items),
            "parameter_count": int(items[0]["parameter_count"]),
        }
        for k in metric_keys:
            vals = np.asarray([float(i[k]) for i in items], dtype=np.float64)
            agg[f"{k}_mean"] = round(float(vals.mean()), 6)
            agg[f"{k}_std"] = round(float(vals.std()), 6) if len(vals) > 1 else 0.0

        times = [float(i["training_time_sec"]) for i in items if i["training_time_sec"] is not None]
        if times:
            t_arr = np.asarray(times, dtype=np.float64)
            agg["training_time_sec_mean"] = round(float(t_arr.mean()), 3)
            agg["training_time_sec_std"] = round(float(t_arr.std()), 3) if len(t_arr) > 1 else 0.0
        else:
            agg["training_time_sec_mean"] = None
            agg["training_time_sec_std"] = None

        summary[model_name] = agg
    return summary


def compare_experiments(
    results_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    output_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Compare completed experiment runs and write summary CSV, JSON, and comparison plot."""
    res_root = Path(results_dir)
    runs = collect_completed_runs(res_root)
    aggregated = aggregate_by_model(runs)

    csv_target = Path(output_csv) if output_csv is not None else (res_root / "comparison_summary.csv")
    json_target = Path(output_json) if output_json is not None else (res_root / "comparison_summary.json")

    if runs:
        csv_target.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_target, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(runs[0].keys()))
            writer.writeheader()
            writer.writerows(runs)

    if "baseline_cnn" in aggregated and "attention_cnn" in aggregated:
        base_m = {
            "accuracy": aggregated["baseline_cnn"]["accuracy_mean"],
            "macro_precision": aggregated["baseline_cnn"]["macro_precision_mean"],
            "macro_recall": aggregated["baseline_cnn"]["macro_recall_mean"],
            "macro_f1": aggregated["baseline_cnn"]["macro_f1_mean"],
        }
        attn_m = {
            "accuracy": aggregated["attention_cnn"]["accuracy_mean"],
            "macro_precision": aggregated["attention_cnn"]["macro_precision_mean"],
            "macro_recall": aggregated["attention_cnn"]["macro_recall_mean"],
            "macro_f1": aggregated["attention_cnn"]["macro_f1_mean"],
        }
        plot_model_comparison(base_m, attn_m, res_root / "model_comparison.png")

    payload = {
        "results_dir": str(res_root),
        "total_runs_found": len(runs),
        "runs": runs,
        "aggregated_by_model": aggregated,
    }
    save_json(payload, json_target)
    return payload
