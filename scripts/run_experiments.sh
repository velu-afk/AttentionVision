#!/usr/bin/env bash
# Controlled Multi-Seed CIFAR-10 Experiment Runner
# Trains and evaluates BaselineCNN and AttentionCNN across 3 independent random seeds (42, 43, 44),
# runs the controlled corruption sensitivity benchmark, and generates the summary comparison table.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python3}"
CONFIG_PATH="${1:-configs/default.yaml}"
SEEDS=(42 43 44)

echo "=== AttentionVision Controlled Multi-Seed Experiment Suite ==="
echo "Config: ${CONFIG_PATH} | Seeds: ${SEEDS[*]}"

"${PYTHON_BIN}" -m attentionvision inspect-env
"${PYTHON_BIN}" -m attentionvision prepare-data --config "${CONFIG_PATH}" --sample-grid-out results/cifar10_sample_grid.png

for SEED in "${SEEDS[@]}"; do
    for MODEL in "baseline_cnn" "attention_cnn"; do
        echo "----------------------------------------------------------------"
        echo "Training model=${MODEL} | seed=${SEED}"
        echo "----------------------------------------------------------------"
        "${PYTHON_BIN}" -m attentionvision train \
            --config "${CONFIG_PATH}" \
            --model "${MODEL}" \
            --seed "${SEED}"

        CKPT="checkpoints/${MODEL}_seed${SEED}/best_model.pt"
        echo "Evaluating checkpoint: ${CKPT}"
        "${PYTHON_BIN}" -m attentionvision evaluate \
            --config "${CONFIG_PATH}" \
            --model "${MODEL}" \
            --seed "${SEED}" \
            --checkpoint "${CKPT}"
    done
done

echo "----------------------------------------------------------------"
echo "Running Controlled Corruption Sensitivity Experiment (Seed 42 Checkpoints)"
echo "----------------------------------------------------------------"
"${PYTHON_BIN}" -m attentionvision robustness \
    --config "${CONFIG_PATH}" \
    --baseline-checkpoint "checkpoints/baseline_cnn_seed42/best_model.pt" \
    --attention-checkpoint "checkpoints/attention_cnn_seed42/best_model.pt" \
    --output-dir "results/robustness"

echo "----------------------------------------------------------------"
echo "Generating Multi-Seed Summary Table and Comparison Plots"
echo "----------------------------------------------------------------"
"${PYTHON_BIN}" scripts/compare_results.py --results-dir results

echo "=== All CIFAR-10 Experiments Completed! ==="
