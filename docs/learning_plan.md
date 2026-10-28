# Two-Day Prioritized Learning & Interview Preparation Plan

This structured 48-hour study plan (~10 focused hours across 2 days) is designed so you can master the **AttentionVision** codebase, understand every mathematical operation, and defend the project confidently in an ML internship interview.

> **Core Rule**: Spend 85% of your time on the models, tensor shapes, forward/backward math, training loop, and experimental design (`Modules 1–8`), and no more than 15% on CLI/CI tooling (`Module 9`).

---

## Day 1: Architectures, Spatial Attention, and the Training Pipeline (~5 Hours)

### Task 1: The CNN Forward Pass & Tensor Shapes
- **Topic**: 3-stage convolutional backbone (`ConvBlock`, `MaxPool2d`, `AdaptiveAvgPool2d`, `Linear`).
- **Recommended Time**: 45 minutes.
- **Exact File to Inspect**: `src/attentionvision/models/cnn.py` (specifically `ConvBlock` and `BaselineCNN`).
- **Concept to Learn**: How input images of shape $(B, 3, 32, 32)$ transform through each stage into $(B, 32, 16, 16) \to (B, 64, 8, 8) \to (B, 128, 4, 4) \to (B, 128) \to (B, 10)$, and why `bias=False` is used before `BatchNorm2d`.
- **Question to Answer Aloud**: *"Starting from a $(4, 3, 32, 32)$ batch, what is the exact tensor shape after every layer in `BaselineCNN`, and why does global average pooling reduce overfitting compared with flattening $(4, 128, 4, 4)$ directly?"*
- **Practical Check to Perform**:
  ```bash
  python3 -c "import torch; from attentionvision.models import BaselineCNN; m = BaselineCNN(); print(m(torch.randn(4, 3, 32, 32)).shape)"
  ```
- **Common Misconception to Avoid**: Thinking that `MaxPool2d(2, 2)` changes the number of channels—it only halves height and width while keeping channel count $C$ unchanged.

---

### Task 2: Convolutions, Feature Maps, and Receptive Fields
- **Topic**: Spatial dimension arithmetic, parameter counting, and effective receptive fields.
- **Recommended Time**: 45 minutes.
- **Exact File to Inspect**: `docs/architecture.md` (Section 2) and `tests/test_models.py`.
- **Concept to Learn**: How to compute output spatial size $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k)/s\rfloor + 1$ and parameter count $C_{\text{out}} \times C_{\text{in}} \times k_H \times k_W$ for every layer ($94,762$ total parameters in `BaselineCNN`).
- **Question to Answer Aloud**: *"Why does `block3.conv` (`Conv2d(64, 128, 3x3)`) have $73,728$ parameters while `block1.conv` (`Conv2d(3, 32, 3x3)`) has only $864$ parameters?"*
- **Practical Check to Perform**:
  ```bash
  pytest tests/test_models.py -v
  ```
- **Common Misconception to Avoid**: Assuming a $3 \times 3$ convolution in Stage 3 only sees a $3 \times 3$ pixel patch of the original image—because of two preceding $2 \times 2$ pooling layers, each Stage 3 unit has an $18 \times 18$ receptive field on the input image.

---

### Task 3: The Spatial-Attention Mechanism (`SpatialAttention`)
- **Topic**: Channel-wise mean/max pooling, $7 \times 7$ spatial convolution, sigmoid gating, and residual broadcasting.
- **Recommended Time**: 75 minutes.
- **Exact File to Inspect**: `src/attentionvision/models/attention.py` and `tests/test_attention.py`.
- **Concept to Learn**:
  $$M_s(F) = \sigma\left(\text{Conv2d}_{2 \to 1, 7 \times 7}\left([\text{Mean}_c(F); \text{Max}_c(F)]\right)\right), \quad F_{\text{out}} = F \odot (1 + M_s(F))$$
- **Question to Answer Aloud**: *"How does `SpatialAttention` compress $C$ channels into $2$ channels with zero learnable parameters, why does the $7 \times 7$ conv use padding $3$, and why do we use $F \odot (1 + M_s(F))$ instead of $F \odot M_s(F)$?"*
- **Practical Check to Perform**:
  ```bash
  pytest tests/test_attention.py -v
  ```
- **Common Misconception to Avoid**: Claiming that `Sigmoid` spatial attention weights sum to $1$ across pixels (they do not—each pixel is independently gated in $(0, 1)$) or claiming that attention maps provide causal explanations of predictions.

---

### Task 4: Cross-Entropy Loss, Backpropagation, and Gradient Flow
- **Topic**: Logits vs. probabilities, backward pass through residual multiplication, and AdamW updates.
- **Recommended Time**: 60 minutes.
- **Exact File to Inspect**: `src/attentionvision/train.py` (`train_one_epoch` and `build_optimizer_and_scheduler`) and `docs/interview_guide.md` (Sections 3–6).
- **Concept to Learn**: Why `nn.CrossEntropyLoss` expects raw unnormalized logits $(B, 10)$, why `optimizer.zero_grad()` must precede `loss.backward()`, and how gradients flow into both the backbone and `attn.conv.weight`.
- **Question to Answer Aloud**: *"What happens mathematically during `loss.backward()` and `optimizer.step()`, and why would forgetting `optimizer.zero_grad()` break training?"*
- **Practical Check to Perform**: Inspect `test_spatial_attention_gradient_propagation` in `tests/test_attention.py` and verify that `attn.conv.weight.grad` has shape `(1, 2, 7, 7)` and finite values.
- **Common Misconception to Avoid**: Applying `F.softmax(logits, dim=-1)` inside `forward()` before passing outputs to `nn.CrossEntropyLoss()` (which applies `LogSoftmax` internally—doing softmax twice distorts gradients!).

---

### Task 5: Training, Validation, and Checkpoint Resumption
- **Topic**: Train vs. eval modes (`BatchNorm2d`, `Dropout`), early stopping, and stateful checkpointing.
- **Recommended Time**: 60 minutes.
- **Exact File to Inspect**: `src/attentionvision/train.py` (`validate_epoch`, `save_checkpoint`, `load_checkpoint`, `run_training`) and `tests/test_checkpoint.py`.
- **Concept to Learn**: Why both `model.eval()` and `with torch.no_grad():` are required during validation, and why saving `optimizer_state_dict` and `scheduler_state_dict` is essential for resuming training seamlessly.
- **Question to Answer Aloud**: *"If I only call `with torch.no_grad():` during validation and forget `model.eval()`, what goes wrong inside `BatchNorm2d` and `Dropout`?"*
- **Practical Check to Perform**:
  ```bash
  pytest tests/test_checkpoint.py tests/test_training.py -v
  ```
- **Common Misconception to Avoid**: Selecting the best checkpoint using test-set accuracy instead of validation-set accuracy.

---

## Day 2: Data Hygiene, Metrics, Fairness, Robustness, and Mock Defense (~5 Hours)

### Task 6: Data Splitting, Transform Isolation, and Evaluation Metrics
- **Topic**: Disjoint train/val splitting, augmentation isolation, macro-F1, confusion matrices, and latency benchmarking.
- **Recommended Time**: 60 minutes.
- **Exact File to Inspect**: `src/attentionvision/data.py`, `src/attentionvision/metrics.py`, and `src/attentionvision/evaluate.py`.
- **Concept to Learn**: Why `train_subset` and `val_subset` wrap two separate dataset instances (`train_base` with random crop/flip vs. `val_base` with deterministic normalization), and how `compute_classification_metrics` and `measure_inference_latency` work.
- **Question to Answer Aloud**: *"Why would calling `random_split` on a single `CIFAR10(transform=train_transform)` object cause subtle data contamination in validation?"*
- **Practical Check to Perform**:
  ```bash
  pytest tests/test_data.py tests/test_metrics.py -v
  ```
- **Common Misconception to Avoid**: Comparing inference latencies measured with different batch sizes or without warm-up iterations.

---

### Task 7: Experimental Fairness and Interpreting Negative/Inconclusive Results
- **Topic**: Controlled ablation design, cross-seed variance ($\text{mean} \pm \text{std}$), and parameter vs. latency trade-offs.
- **Recommended Time**: 45 minutes.
- **Exact File to Inspect**: `docs/methodology.md`, `docs/experiments.md`, and `scripts/compare_results.py`.
- **Concept to Learn**: How holding the backbone, split seed, optimizer, and schedule constant isolates the effect of spatial attention, and how to judge whether an accuracy difference exceeds cross-seed standard deviation.
- **Question to Answer Aloud**: *"Why is a negative or inconclusive result on CIFAR-10 still scientifically valuable, and how do I explain the $+4.7\%$ latency overhead when parameter count only increased by $0.21\%$?"*
- **Practical Check to Perform**:
  ```bash
  python3 scripts/compare_results.py --results-dir results/smoke
  ```
- **Common Misconception to Avoid**: Treating a $+0.1\%$ single-seed difference as proof that attention is superior without checking multi-seed variance.

---

### Task 8: Corruption Sensitivity Experiment (`robustness.py`)
- **Topic**: Evaluation-time Gaussian noise and Gaussian blur sweeps.
- **Recommended Time**: 45 minutes.
- **Exact File to Inspect**: `src/attentionvision/robustness.py` and `results/smoke/robustness/robustness_results.csv`.
- **Concept to Learn**: How `apply_gaussian_noise` and `apply_gaussian_blur` perturb test batches deterministically using `corruption_seed=1337`, and why spatial attention can help under mild blur ($\sigma = 0.5$) yet degrade under severe blur ($\sigma \ge 1.0$).
- **Question to Answer Aloud**: *"Why did we apply corruptions only during evaluation rather than during training, and why doesn't this experiment prove general real-world out-of-distribution robustness?"*
- **Practical Check to Perform**: Inspect `results/smoke/robustness/robustness_gaussian_blur.png` and `results/smoke/robustness/robustness_gaussian_noise.png`.
- **Common Misconception to Avoid**: Calling synthetic Gaussian noise/blur testing "general domain-shift robustness."

---

### Task 9: Repository Structure, Reproducibility, and Full Mock Defense
- **Topic**: Configuration management (`config.py`), CLI (`cli.py`), limitations (`docs/limitations.md`), and aloud Q&A practice.
- **Recommended Time**: 90 minutes.
- **Exact File to Inspect**: `docs/interview_defense.md`, `docs/limitations.md`, and `src/attentionvision/config.py`.
- **Concept to Learn**: Practice delivering the 30-second pitch, the 2-minute technical walkthrough, and answering 15 random questions from `docs/interview_defense.md` aloud without looking at notes.
- **Question to Answer Aloud**: *"Walk me through AttentionVision from raw image tensor to final evaluation and tell me its three biggest limitations."*
- **Practical Check to Perform**: Run the full smoke pipeline end-to-end and inspect the generated plots:
  ```bash
  bash scripts/run_smoke_test.sh
  ```
- **Common Misconception to Avoid**: Claiming bitwise reproducibility across different hardware architectures (CPU vs. GPU).
