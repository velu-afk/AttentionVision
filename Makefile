.PHONY: help install env test lint format smoke train-baseline train-attention evaluate robustness compare clean

PYTHON ?= python3
PYTEST ?= pytest
RUFF ?= ruff

help:
	@echo "AttentionVision Makefile Commands:"
	@echo "  make install          Install package in editable mode with dev dependencies"
	@echo "  make env              Inspect Python, PyTorch, CPU/RAM, and CUDA availability"
	@echo "  make test             Run offline unit test suite via pytest"
	@echo "  make lint             Run Ruff linter checks"
	@echo "  make format           Check code formatting with Ruff"
	@echo "  make smoke            Run rapid end-to-end smoke test for both models"
	@echo "  make train-baseline   Train Baseline CNN using configs/default.yaml"
	@echo "  make train-attention  Train Attention CNN using configs/default.yaml"
	@echo "  make evaluate         Evaluate trained checkpoints and generate plots"
	@echo "  make robustness       Run corruption sensitivity benchmark"
	@echo "  make compare          Aggregate and compare completed experiment runs"
	@echo "  make clean            Remove temporary caches and build artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

env:
	$(PYTHON) -m attentionvision inspect-env

test:
	$(PYTEST) tests

lint:
	$(RUFF) check src tests scripts

format:
	$(RUFF) format --check src tests scripts

smoke:
	bash scripts/run_smoke_test.sh

train-baseline:
	$(PYTHON) -m attentionvision train --config configs/default.yaml --model baseline_cnn

train-attention:
	$(PYTHON) -m attentionvision train --config configs/default.yaml --model attention_cnn

evaluate:
	$(PYTHON) -m attentionvision evaluate --config configs/default.yaml --checkpoint checkpoints/baseline_cnn_seed42/best_model.pt
	$(PYTHON) -m attentionvision evaluate --config configs/default.yaml --checkpoint checkpoints/attention_cnn_seed42/best_model.pt

robustness:
	$(PYTHON) -m attentionvision robustness --config configs/default.yaml \
		--baseline-checkpoint checkpoints/baseline_cnn_seed42/best_model.pt \
		--attention-checkpoint checkpoints/attention_cnn_seed42/best_model.pt

compare:
	$(PYTHON) scripts/compare_results.py --results-dir results

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist src/*.egg-info
