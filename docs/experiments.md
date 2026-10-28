# Experiment Management, Templates, and Results Interpretation

## 1. Classification of Experiment Run Types

Every configuration in `AttentionVision` explicitly sets `run_type` in YAML (`configs/smoke.yaml` or `configs/default.yaml`) so that smoke checks and final benchmarks are never conflated:

| `run_type` | Purpose | Dataset | Typical Budget |
| :--- | :--- | :--- | :--- |
| `smoke` | Fast end-to-end verification of training, checkpointing, evaluation, and plotting on CPU without network downloads. | Procedural `SyntheticCIFAR10Dataset` (`use_synthetic: true`) | $3\text{--}5$ epochs, $240$ train/val samples |
| `debug` | Diagnosing gradient flow, learning rates, or single-batch overfitting. | Synthetic or small CIFAR-10 subset | $1\text{--}5$ epochs |
| `exploratory` | Validation-only checks of architectural or hyperparameter settings (never evaluated on the test set). | Official CIFAR-10 train/val split ($45\text{k}/5\text{k}$) | $10\text{--}15$ epochs |
| `final_evaluation` | Controlled single-seed comparison on official CIFAR-10 using locked hyperparameters. | Official CIFAR-10 ($45\text{k}$ train / $5\text{k}$ val / $10\text{k}$ test) | $30$ epochs |
| `multi_seed` | Repeated independent seeds (`42, 43, 44`) reporting $\text{mean} \pm \text{std}$. | Official CIFAR-10 ($45\text{k}$ train / $5\text{k}$ val / $10\text{k}$ test) | $30$ epochs $\times 3$ seeds |

## 2. Completed Sandbox Verification Experiments (3-Seed Smoke Benchmark)

In the offline CPU build environment (Linux 6.12 x86_64, 2 CPU cores, 7.76 GiB RAM, no GPU/CUDA, no outbound network access), we executed a 3-seed (`seeds 42, 43, 44`) controlled smoke benchmark (`configs/smoke.yaml`, 5 epochs per run) plus the 18-condition robustness sweep (`results/smoke/`):

### 2.1 Multi-Seed Smoke Comparison (`results/smoke/comparison_table.md`)

| Model | Runs (Seeds) | Data Type | Params | Val Acc (%) | Test Acc (%) | Final Val Loss (Seed 42) | Train Time (s) | Latency (ms/img) |
| :--- | :--- | :--- | ---: | :--- | :--- | ---: | :--- | ---: |
| `baseline_cnn` | 3 (`42, 43, 44`) | Synthetic (Smoke) | $94,762$ | $100.00 \pm 0.00$ | $100.00 \pm 0.00$ | $0.3736$ | $16.77 \pm 0.28$ | $5.558$ |
| `attention_cnn` | 3 (`42, 43, 44`) | Synthetic (Smoke) | $94,958$ | $100.00 \pm 0.00$ | $100.00 \pm 0.00$ | $0.1501$ | $18.32 \pm 0.54$ | $5.821$ |

### 2.2 Smoke Corruption Sensitivity Results (`results/smoke/robustness/robustness_results.csv`, Seed 42)

| Corruption Type | Severity ($\sigma$) | `BaselineCNN` Acc (%) | `AttentionCNN` Acc (%) | `BaselineCNN` Macro-F1 (%) | `AttentionCNN` Macro-F1 (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Clean Reference** | $0.00$ | $100.00$ | $100.00$ | $100.00$ | $100.00$ |
| **Gaussian Noise** | $0.25$ | $100.00$ | $100.00$ | $100.00$ | $100.00$ |
| **Gaussian Noise** | $0.50$ | $50.00$ | $50.00$ | $38.94$ | $40.00$ |
| **Gaussian Noise** | $0.75$ | $30.00$ | $28.33$ | $17.86$ | $21.27$ |
| **Gaussian Noise** | $1.00$ | $20.00$ | $16.67$ | $8.07$ | $11.94$ |
| **Gaussian Blur** | $0.50$ | $68.33$ | $83.33$ | $62.07$ | $80.07$ |
| **Gaussian Blur** | $1.00$ | $45.00$ | $20.00$ | $36.14$ | $12.00$ |
| **Gaussian Blur** | $1.50$ | $38.33$ | $20.00$ | $30.64$ | $12.00$ |

**Empirical Observation on Smoke Benchmark**:
1. Both models reach $100\%$ clean accuracy on the localized geometric patterns in `SyntheticCIFAR10Dataset`, with `AttentionCNN` achieving a lower validation cross-entropy loss ($0.1501$ vs. $0.3736$) at the cost of $+196$ parameters and $+4.7\%$ CPU inference latency ($5.821\text{ ms}$ vs. $5.558\text{ ms}$).
2. Under **mild Gaussian blur ($\sigma = 0.5$)**, `AttentionCNN` retains higher accuracy ($83.33\%$ vs. $68.33\%$), whereas under **severe Gaussian blur ($\sigma \ge 1.0$)**, `AttentionCNN` degrades more sharply ($20.00\%$ vs. $45.00\%$). Importantly, because this smoke experiment uses synthetic procedural patterns, these numbers verify pipeline mechanics and illustrate Hypothesis H2 in a toy setting—they do **not** substitute for the full 50,000-image CIFAR-10 benchmark below.

## 3. Official CIFAR-10 Results Template (Target Laptop Execution)

When running `bash scripts/run_experiments.sh` on your Lenovo LOQ (NVIDIA RTX 4050 GPU or CPU with internet access to download CIFAR-10), `scripts/compare_results.py --results-dir results` automatically populates `results/comparison_table.md` and `results/comparison_summary.csv`:

| Model | Seeds | Dataset | Params | Best Val Acc (%) | Test Acc (%) | Macro Precision (%) | Macro Recall (%) | Macro-F1 (%) | Train Time (s) | Latency (ms/img) |
| :--- | :--- | :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_cnn` | `42, 43, 44` | Official CIFAR-10 | $94,762$ | *Pending (`run_experiments.sh`)* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `attention_cnn` | `42, 43, 44` | Official CIFAR-10 | $94,958$ | *Pending (`run_experiments.sh`)* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

### Estimated Full-Training Cost on Target Hardware
- **NVIDIA RTX 4050 Laptop GPU (CUDA)**: ~4–6 seconds per epoch ($45,000$ training images, `batch_size=128`), or **~2–3 minutes per 30-epoch run** (~15–18 minutes for all 6 runs across 3 seeds).
- **Modern Laptop CPU (8+ threads)**: ~35–60 seconds per epoch, or **~18–30 minutes per 30-epoch run**.

## 4. Procedure for Interpreting Negative or Inconclusive Findings

When evaluating the completed CIFAR-10 table:
1. **Compare Difference Against Cross-Seed Standard Deviation**: If `AttentionCNN` improves mean test accuracy by $\Delta = 0.25\%$ but the standard deviation across seeds `42, 43, 44` is $\pm 0.40\%$, the improvement is within experimental noise and must be reported as **inconclusive**.
2. **Check Per-Class Trade-offs**: Inspect `results/baseline_cnn_seed42/per_class_metrics.csv` and `results/attention_cnn_seed42/per_class_metrics.csv`. Does attention improve fine-grained animal classes (`cat`, `dog`, `bird`) where foreground localization matters, while leaving background-heavy classes (`ship`, `airplane`) unchanged?
3. **Evaluate Cost vs. Gain**: Even a statistically consistent $+0.5\%$ accuracy gain must be weighed against the $+4\text{--}8\%$ latency overhead from channel-pooling and sigmoid gating.
4. **Report Negative Results Honestly**: A well-controlled negative finding ("Adding spatial attention at $16 \times 16$ and $8 \times 8$ resolutions on $32 \times 32$ CIFAR-10 images did not yield a statistically significant test accuracy gain over the 3-stage CNN baseline") is a completely valid and scientifically rigorous research conclusion.
