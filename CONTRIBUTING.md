# Contributing to AttentionVision

Thank you for your interest in contributing to **AttentionVision**. Because this repository is designed as a controlled, reproducible empirical study comparing a compact CNN baseline against a spatial-attention CNN on CIFAR-10, contributions must preserve experimental fairness, simplicity, and testability.

## Core Principles

1. **Controlled Experimental Fairness**: Any change to the backbone architecture, data preprocessing, optimizer, or evaluation protocol must apply equally to both `BaselineCNN` and `AttentionCNN` unless the change specifically targets an isolated ablation study.
2. **Offline Unit Test Independence**: Unit tests inside `tests/` must execute on CPU without downloading external datasets (use synthetic tensors or temporary fixtures).
3. **Beginner-Understandable Code**: Favor explicit, readable PyTorch modules with clear tensor-shape comments over deeply nested abstractions or metaprogramming.
4. **Honest Reporting**: Never commit fabricated metrics, cherry-picked test seeds, or unverified claims.

## Development Workflow

1. Create a virtual environment and install the package in editable mode:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. Run the automated test suite and linter before submitting changes:
   ```bash
   pytest tests
   ruff check src tests scripts
   ruff format --check src tests scripts
   ```
3. Run the smoke test script to verify the end-to-end training, evaluation, and robustness pipeline:
   ```bash
   bash scripts/run_smoke_test.sh
   ```
4. Update documentation (`README.md` and `docs/`) whenever CLI flags, YAML configuration keys, or model tensor dimensions change.
