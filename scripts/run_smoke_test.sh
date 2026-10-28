#!/usr/bin/env bash
# Rapid CPU Smoke Test Script for AttentionVision
# Trains and evaluates both BaselineCNN and AttentionCNN on synthetic 32x32 RGB data
# (no GPU or internet download required), runs corruption sensitivity checks, and compares outputs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python3}"
CONFIG_PATH="${1:-configs/smoke.yaml}"

echo "=== [1/6] Inspecting Runtime Environment ==="
"${PYTHON_BIN}" -m attentionvision inspect-env

echo "=== [2/6] Preparing & Verifying Smoke Dataset ==="
"${PYTHON_BIN}" -m attentionvision prepare-data --config "${CONFIG_PATH}" --sample-grid-out results/smoke/sample_grid.png

echo "=== [3/6] Training BaselineCNN (Smoke Config) ==="
"${PYTHON_BIN}" -m attentionvision train --config "${CONFIG_PATH}" --model baseline_cnn

echo "=== [4/6] Training AttentionCNN (Smoke Config) ==="
"${PYTHON_BIN}" -m attentionvision train --config "${CONFIG_PATH}" --model attention_cnn

echo "=== [5/6] Evaluating Checkpoints & Running Robustness Sweep ==="
"${PYTHON_BIN}" -m attentionvision evaluate --config "${CONFIG_PATH}" \
    --model baseline_cnn \
    --checkpoint checkpoints/smoke/baseline_cnn_seed42/best_model.pt

"${PYTHON_BIN}" -m attentionvision evaluate --config "${CONFIG_PATH}" \
    --model attention_cnn \
    --checkpoint checkpoints/smoke/attention_cnn_seed42/best_model.pt

"${PYTHON_BIN}" -m attentionvision robustness --config "${CONFIG_PATH}" \
    --baseline-checkpoint checkpoints/smoke/baseline_cnn_seed42/best_model.pt \
    --attention-checkpoint checkpoints/smoke/attention_cnn_seed42/best_model.pt \
    --output-dir results/smoke/robustness

echo "=== [6/6] Aggregating & Comparing Smoke Experiment Results ==="
"${PYTHON_BIN}" scripts/compare_results.py --results-dir results/smoke

echo "=== Smoke Test Completed Successfully! Outputs saved under results/smoke/ ==="
