# Experimental Methodology

## 1. Research Question

Does adding a lightweight spatial-attention mechanism to a compact convolutional neural network (CNN) improve image-classification performance, alter sensitivity to synthetic image corruptions, or introduce measurable computational overhead compared with an otherwise identical CNN baseline on $32 \times 32$ RGB images?

## 2. Research Hypotheses

All claims in this repository are treated as testable empirical hypotheses rather than assumed truths:

- **Hypothesis H1 (Clean Classification Performance)**: Adding spatial attention after intermediate convolutional blocks (`Stage 2` at $16 \times 16$ and `Stage 3` at $8 \times 8$) may improve validation and test accuracy or speed up loss convergence by allowing the network to emphasize informative spatial regions prior to pooling.
- **Hypothesis H2 (Corruption Sensitivity)**: Spatial attention may either improve or worsen classification accuracy under synthetic pixel perturbations (additive Gaussian noise and spatial Gaussian blur). For example, spatial gating may suppress mild background blur yet fail when severe blur erodes the local activation peaks that drive the attention map.
- **Hypothesis H3 (Efficiency and Variability Trade-off)**: Any accuracy or robustness difference must be weighed against parameter overhead ($+196$ parameters for two $7 \times 7$ spatial-attention modules), training duration, single-sample inference latency, and cross-seed variance.

## 3. Dataset and Splitting Protocol

### 3.1 Benchmark Dataset: CIFAR-10
- **Source**: Alex Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*.
- **Total Size**: $60,000$ color images ($3 \times 32 \times 32$ pixels) uniformly distributed across $10$ mutually exclusive object classes (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).
- **Official Partitions**: $50,000$ official training images and $10,000$ official test images.

### 3.2 Deterministic Train / Validation Partition
To prevent test-set leakage, the $50,000$ official training images are partitioned deterministically using `compute_train_val_indices(total_size=50000, val_fraction=0.10, split_seed=42)` in `src/attentionvision/data.py`:
- **Training Split**: $45,000$ images ($90\%$ of official training set).
- **Validation Split**: $5,000$ images ($10\%$ of official training set).
- **Test Split**: $10,000$ official test images (held out strictly for final evaluation and corruption sensitivity sweeps).

### 3.3 Data Hygiene and Transform Isolation
A common beginner bug in PyTorch is applying `torch.utils.data.random_split` to a single `CIFAR10` instance configured with training augmentations, which either leaks stochastic augmentations into validation or strips augmentations from training if overwritten. In `src/attentionvision/data.py`:
1. Two separate dataset objects (`train_base` and `val_base`) are instantiated over the same underlying data.
2. `train_base` is wrapped with `build_transforms(data_cfg, is_train=True)`:
   - `RandomCrop(32, padding=4)`
   - `RandomHorizontalFlip(p=0.5)`
   - `ToTensor()` (scales uint8 $[0, 255]$ to float32 $[0.0, 1.0]$)
   - `Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])`
3. `val_base` and `test_dataset` are wrapped with `build_transforms(data_cfg, is_train=False)`:
   - `ToTensor()`
   - `Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])`
4. Disjoint index lists (`train_indices` and `val_indices`) are passed to `Subset(train_base, train_indices)` and `Subset(val_base, val_indices)`. Unit tests in `tests/test_data.py` verify both index disjointness and validation determinism.

## 4. Controlled Model Comparison

To isolate the causal effect of the spatial-attention module:
- **Model A (`BaselineCNN`)** and **Model B (`AttentionCNN`)** share the exact same 3-stage convolutional backbone ($3 \to 32 \to 64 \to 128$ channels), batch normalization layers, ReLU activations, $2 \times 2$ max-pooling layers, adaptive global average pooling, dropout rate ($p = 0.25$), and linear classification head ($128 \to 10$).
- Both models are trained with the exact same `split_seed`, optimizer (`AdamW`), initial learning rate ($10^{-3}$), weight decay ($5 \times 10^{-4}$), learning-rate schedule (`CosineAnnealingLR`), batch size ($128$), and epoch budget ($30$ epochs in `configs/default.yaml`).
- Checkpoint selection for both models uses the exact same validation criterion: highest validation accuracy (with lowest validation cross-entropy loss as tie-breaker).

## 5. Evaluation Metrics and Latency Protocol

### 5.1 Classification Metrics
Implemented in `src/attentionvision/metrics.py`:
- **Overall Accuracy**: $\frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)$
- **Per-Class Precision ($P_c$), Recall ($R_c$), and F1 ($F_{1,c}$)**:
  $$P_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad R_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}, \quad F_{1,c} = \frac{2 P_c R_c}{P_c + R_c}$$
- **Macro-Averaged Precision, Recall, and F1**: Unweighted arithmetic mean across all $C = 10$ classes.
- **Confusion Matrix**: $10 \times 10$ integer matrix $M_{i,j}$ counting samples of true class $i$ predicted as class $j$.

### 5.2 Inference Latency Protocol
Latency measurements depend strongly on hardware, batch size, and thermal state. `measure_inference_latency` records:
- Execution device (`cpu` or `cuda:0`) and input shape `(batch_size, 3, 32, 32)` (default single-sample latency `batch_size=1`).
- `warmup_runs` untimed forward passes under `model.eval()` and `torch.no_grad()` (with `torch.cuda.synchronize()` when CUDA is active).
- `measure_runs` timed forward passes reporting mean and standard deviation in milliseconds per batch, milliseconds per sample, and throughput in samples per second.

## 6. Controlled Corruption Sensitivity Experiment

Implemented in `src/attentionvision/robustness.py`:
- Corruptions are applied **exclusively at evaluation time** on the test split using a fixed `corruption_seed=1337` so both models receive identical perturbed tensors:
  1. **Clean Reference (`sigma = 0.0`)**: Unperturbed normalized test images.
  2. **Additive Gaussian Noise**: $x_{\text{noisy}} = x + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ for $\sigma \in \{0.0, 0.15, 0.30, 0.50, 0.75, 1.00\}$ (or $\{0.0, 0.25, 0.50, 0.75, 1.00\}$ in `configs/smoke.yaml`).
  3. **Spatial Gaussian Blur**: 2D spatial convolution with a Gaussian kernel ($k = 5$) across spatial standard deviations $\sigma \in \{0.0, 0.5, 1.0, 1.5, 2.0\}$.
- **Scope Limitation**: These experiments measure sensitivity to specific parametric pixel perturbations and do not establish general real-world out-of-distribution (OOD) robustness.

## 7. Reproducibility Controls

- `set_seed(seed, deterministic=True)` seeds Python `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and sets `cudnn.deterministic = True` and `cudnn.benchmark = False`.
- Every run saves its resolved YAML configuration (`config.yaml`), software/hardware environment metadata (`training_summary.json`), epoch-by-epoch metrics (`history.csv`), and evaluation outputs (`evaluation_metrics.json`, `per_class_metrics.csv`) in a dedicated run folder (`results/<model_name>_seed<seed>/`).
