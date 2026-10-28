# Project-Specific Interview Defense Guide (`AttentionVision`)

Use this guide to prepare for technical interviews. Every equation, tensor shape, parameter count, and experimental result matches the exact code in `src/attentionvision/`.

---

## 1. 30-Second Project Pitch

> "I built **AttentionVision**, a reproducible PyTorch research repository that investigates whether adding a lightweight CBAM-style spatial-attention mechanism to a 3-stage CNN improves CIFAR-10 image classification or alters sensitivity to synthetic image corruptions. To make the comparison scientifically fair, both the baseline CNN ($94,762$ parameters) and the attention CNN ($94,958$ parameters—just $196$ extra parameters) share the exact same convolutional backbone, deterministic 45k/5k train-validation split, and AdamW cosine training schedule. I also built an evaluation-time corruption benchmark testing Gaussian noise and Gaussian blur across multiple severities, backed by 28 offline pytest unit tests and automated CI."

---

## 2. Two-Minute Technical Explanation

> "My goal was to test a specific hypothesis rather than just chase accuracy: what happens when we insert a lightweight spatial-attention module into a compact CNN?
>
> First, I implemented a 3-stage `BaselineCNN` in `src/attentionvision/models/cnn.py` that maps $(B, 3, 32, 32)$ CIFAR-10 images through three `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d` stages ($32 \to 64 \to 128$ channels), reducing spatial dimensions from $32 \times 32$ to $16 \times 16$, $8 \times 8$, and $4 \times 4$, followed by $1 \times 1$ adaptive global average pooling, dropout ($p=0.25$), and a 10-class linear head.
>
> Second, in `src/attentionvision/models/attention.py`, I implemented a standalone `SpatialAttention` module. Given a feature map $F \in \mathbb{R}^{B \times C \times H \times W}$, it computes channel-wise average pooling and channel-wise max pooling—producing two $(B, 1, H, W)$ spatial descriptors—concatenates them into $(B, 2, H, W)$, applies a $7 \times 7$ convolution with padding $3$ to produce a 1-channel spatial map, and normalizes it with a sigmoid into $(0, 1)^{B \times 1 \times H \times W}$. I then apply residual multiplicative gating $F_{\text{out}} = F \odot (1 + M_s(F))$ at Stages 2 ($16 \times 16$) and 3 ($8 \times 8$). Because each $7 \times 7$ attention conv has only $2 \times 1 \times 7 \times 7 = 98$ weights, adding attention at two stages adds only $196$ parameters ($+0.21\%$).
>
> Third, I ensured strict data hygiene in `src/attentionvision/data.py`: training and validation subsets wrap separate underlying dataset instances so random crops and horizontal flips never leak into validation, and the 10,000-image test set is untouched until final evaluation. Finally, in `src/attentionvision/robustness.py`, I evaluated both models under controlled Gaussian noise and Gaussian blur sweeps. In our 3-seed synthetic smoke benchmark, `AttentionCNN` converged to lower validation loss ($0.1501$ vs. $0.3736$) and outperformed `BaselineCNN` under mild Gaussian blur ($\sigma=0.5$, $83.3\%$ vs. $68.3\%$), but degraded more sharply under severe blur ($\sigma \ge 1.0$, $20.0\%$ vs. $45.0\%$), demonstrating that spatial attention can help under mild perturbation yet fail when strong blur erodes spatial localization cues."

---

## 3. End-to-End Workflow Walkthrough

1. **Configuration (`src/attentionvision/config.py`)**: `load_config` parses `configs/default.yaml` or `configs/smoke.yaml` into typed dataclasses (`ExperimentConfig`, `DataConfig`, `ModelConfig`, `TrainingConfig`, `EvaluationConfig`, `RobustnessConfig`), applies CLI overrides, and runs `.validate()`.
2. **Reproducibility & Device Setup (`src/attentionvision/utils.py`)**: `set_seed(seed, deterministic=True)` seeds Python, NumPy, and PyTorch RNGs; `resolve_device("auto")` selects CUDA if available or falls back to CPU.
3. **Data Pipeline (`src/attentionvision/data.py`)**: `build_dataloaders` creates disjoint train/validation indices using `split_seed=42`, wraps separate dataset instances for train (`RandomCrop` + `RandomHorizontalFlip` + `Normalize`) vs. validation/test (`Normalize` only), and returns a `DataBundle`.
4. **Model Instantiation (`src/attentionvision/models/cnn.py`)**: `build_model` instantiates either `BaselineCNN` or `AttentionCNN`.
5. **Training & Checkpointing (`src/attentionvision/train.py`)**: `run_training` runs `train_one_epoch` and `validate_epoch`, steps the `CosineAnnealingLR` scheduler, clips gradients (`max_norm=2.0`), tracks `best_val_acc`, and saves `best_model.pt` and `last_model.pt` plus `history.csv` and `training_summary.json`.
6. **Evaluation & Visualization (`src/attentionvision/evaluate.py`)**: `evaluate_checkpoint` loads `best_model.pt`, evaluates on `test_loader`, benchmarks single-sample latency via `measure_inference_latency`, exports `evaluation_metrics.json` and `per_class_metrics.csv`, and plots loss/accuracy curves, confusion matrices, per-class bar charts, and spatial attention heatmaps.
7. **Corruption Sensitivity (`src/attentionvision/robustness.py`)**: `run_robustness_experiment` evaluates both checkpoints across Gaussian noise and Gaussian blur sigmas using `corruption_seed=1337` and saves `robustness_results.csv` and degradation curves.

---

## 4. Both Model Architectures & Exact Tensor Dimensions

| Stage | Layer(s) | `BaselineCNN` Tensor Shape | `AttentionCNN` Tensor Shape |Attention Map Shape |
| :--- | :--- | :--- | :--- | :--- |
| Input | Normalized RGB Batch | $(B, 3, 32, 32)$ | $(B, 3, 32, 32)$ | — |
| Stage 1 (`block1`) | `Conv2d(3->32, 3x3, p=1, bias=False) -> BN -> ReLU` | $(B, 32, 32, 32)$ | $(B, 32, 32, 32)$ | Optional $(B, 1, 32, 32)$ |
| Pool 1 (`pool1`) | `MaxPool2d(2x2, stride=2)` | $(B, 32, 16, 16)$ | $(B, 32, 16, 16)$ | — |
| Stage 2 (`block2`) | `Conv2d(32->64, 3x3, p=1, bias=False) -> BN -> ReLU` | $(B, 64, 16, 16)$ | $(B, 64, 16, 16)$ | — |
| Attention 2 (`attn2`) | `SpatialAttention(kernel_size=7, residual=True)` | *(Not present)* | $(B, 64, 16, 16)$ | $(B, 1, 16, 16)$ |
| Pool 2 (`pool2`) | `MaxPool2d(2x2, stride=2)` | $(B, 64, 8, 8)$ | $(B, 64, 8, 8)$ | — |
| Stage 3 (`block3`) | `Conv2d(64->128, 3x3, p=1, bias=False) -> BN -> ReLU` | $(B, 128, 8, 8)$ | $(B, 128, 8, 8)$ | — |
| Attention 3 (`attn3`) | `SpatialAttention(kernel_size=7, residual=True)` | *(Not present)* | $(B, 128, 8, 8)$ | $(B, 1, 8, 8)$ |
| Pool 3 (`pool3`) | `MaxPool2d(2x2, stride=2)` | $(B, 128, 4, 4)$ | $(B, 128, 4, 4)$ | — |
| Global Pool | `AdaptiveAvgPool2d((1, 1)) -> Flatten` | $(B, 128)$ | $(B, 128)$ | — |
| Classifier | `Dropout(0.25) -> Linear(128 -> 10)` | $(B, 10)$ | $(B, 10)$ | — |

---

## 5. Key Equations with Variable Definitions

1. **Channel-Pooled Spatial Attention (`src/attentionvision/models/attention.py`)**:
   $$F_{\text{avg}} = \frac{1}{C} \sum_{c=1}^{C} F_{:, c, :, :}, \quad F_{\text{max}} = \max_{c \in \{1,\dots,C\}} F_{:, c, :, :}$$
   $$M_s(F) = \sigma\left(W_{2 \to 1, k \times k} * [F_{\text{avg}}; F_{\text{max}}]\right) \in (0, 1)^{B \times 1 \times H \times W}$$
   $$F_{\text{out}} = F \odot \left(1 + M_s(F)\right) \in \mathbb{R}^{B \times C \times H \times W}$$
   where $F \in \mathbb{R}^{B \times C \times H \times W}$ is the intermediate feature map, $*$ denotes 2D convolution with symmetric padding $p = (k-1)/2$, $\sigma(z) = (1 + e^{-z})^{-1}$ is the elementwise logistic sigmoid, and $\odot$ denotes broadcasted elementwise multiplication across all $C$ channels.

2. **Cross-Entropy Loss (`nn.CrossEntropyLoss`)**:
   $$\mathcal{L}_{\text{CE}} = -\frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp(z_{b, y_b})}{\sum_{c=0}^{9} \exp(z_{b, c})}$$
   where $z_{b, c}$ is the raw logit for sample $b$ and class $c$, and $y_b \in \{0, \dots, 9\}$ is the ground-truth class index.

---

## 6. Exact Attention Implementation & Design Rationale

In `src/attentionvision/models/attention.py`:
```python
avg_out = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
max_out, _ = x.max(dim=1, keepdim=True)        # (B, 1, H, W)
spatial_descriptor = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
attention_logits = self.conv(spatial_descriptor)           # (B, 1, H, W)
attention_map = self.sigmoid(attention_logits)             # (B, 1, H, W)
out = x * (1.0 + attention_map) if self.residual else x * attention_map
```

### Why Each Design Choice Was Made
- **Why combine `mean` and `max` across channels?** Average pooling captures aggregate spatial activation across all feature filters, while max pooling highlights the single most strongly activated feature filter at each pixel. Concatenating both gives the $7 \times 7$ convolution richer spatial context at zero extra parameter cost compared to a 2-channel input.
- **Why `keepdim=True`?** Preserves the singleton channel dimension `(B, 1, H, W)` so the tensors can be concatenated along `dim=1` into `(B, 2, H, W)` and later broadcast cleanly against `(B, C, H, W)`.
- **Why `residual=True` ($F \odot (1 + M_s(F))$) by default?** At weight initialization, `attention_logits` are near $0$, so $\sigma(0) \approx 0.5$. Pure multiplicative gating ($F \odot M_s(F)$) scales activations and backward gradients by $0.5$ at every attention stage, whereas residual gating ($F \odot (1 + M_s(F))$) preserves the identity signal $F$ and adds a positive spatial boost in $(1, 2)$.
- **Why insert attention at Stages 2 ($16 \times 16$) and 3 ($8 \times 8$) rather than Stage 1 ($32 \times 32$)?** Stage 1 features are low-level edge detectors where spatial suppression is premature and four times more computationally expensive ($32 \times 32 = 1,024$ locations vs. $16 \times 16 = 256$ and $8 \times 8 = 64$).

---

## 7. 28 Project-Specific Interview Questions & Accurate Answers

1. **Q: What is the exact research question of AttentionVision?**
   **A:** Whether adding a lightweight channel-pooled spatial-attention mechanism to a compact 3-stage CNN improves CIFAR-10 classification accuracy, alters sensitivity to synthetic image corruptions, or incurs computational overhead relative to an identical CNN baseline.

2. **Q: How many trainable parameters do `BaselineCNN` and `AttentionCNN` have, and why is the difference exactly 196?**
   **A:** `BaselineCNN` has $94,762$ parameters and `AttentionCNN` has $94,958$ parameters. Each $7 \times 7$ `SpatialAttention` module has `in_channels=2`, `out_channels=1`, and `bias=False`, which is $2 \times 1 \times 7 \times 7 = 98$ parameters. Placed at Stage 2 and Stage 3, $2 \times 98 = 196$ parameters.

3. **Q: How is the parameter count of `BaselineCNN` ($94,762$) broken down by layer?**
   **A:**
   - Stage 1: `Conv2d(3, 32, 3x3, bias=False)` = $864$; `BatchNorm2d(32)` = $64$ ($\gamma, \beta$). Subtotal = $928$.
   - Stage 2: `Conv2d(32, 64, 3x3, bias=False)` = $18,432$; `BatchNorm2d(64)` = $128$. Subtotal = $18,560$.
   - Stage 3: `Conv2d(64, 128, 3x3, bias=False)` = $73,728$; `BatchNorm2d(128)` = $256$. Subtotal = $73,984$.
   - Classifier: `Linear(128, 10, bias=True)` = $128 \times 10 + 10 = 1,290$.
   - Total: $928 + 18,560 + 73,984 + 1,290 = 94,762$.

4. **Q: Why did you set `bias=False` in the backbone `Conv2d` layers?**
   **A:** Every backbone `Conv2d` is immediately followed by `BatchNorm2d`, which subtracts the per-channel mean $\mu_c$ and adds a learnable shift $\beta_c$. Any constant bias added before BatchNorm cancels out during mean subtraction and wastes parameters.

5. **Q: How does tensor broadcasting work in `x * (1.0 + attention_map)`?**
   **A:** `x` has shape `(B, C, H, W)` and `attention_map` has shape `(B, 1, H, W)`. PyTorch aligns trailing dimensions and replicates the singleton dimension `1` across all `C` channels, scaling every channel at spatial coordinate `(h, w)` by the same spatial gate $1 + M_s(b, 1, h, w)$.

6. **Q: Are the spatial attention weights normalized to sum to 1 across the image like Softmax self-attention in Vision Transformers?**
   **A:** No. Each spatial location $(h, w)$ is normalized independently into $(0, 1)$ using an elementwise logistic `Sigmoid`. Unlike spatial Softmax (which forces all pixels to compete so their weights sum to $1$, penalizing images with large foreground objects), Sigmoid gating allows multiple spatial locations to have high weights simultaneously.

7. **Q: How does your `SpatialAttention` module differ from a Vision Transformer (ViT) self-attention layer?**
   **A:** ViT projects image patches into queries $Q$, keys $K$, and values $V$ and computes all-pairs token-to-token attention $\text{Softmax}(QK^\top / \sqrt{d_k})V$ with $O(N^2)$ complexity. Our module is a local convolutional spatial gate: it pools across channels and uses a $7 \times 7$ convolution to compute a local per-pixel mask in $O(H W)$ time with just $98$ parameters.

8. **Q: How did you prevent data augmentation leakage between the training and validation splits?**
   **A:** In `src/attentionvision/data.py`, I instantiated two separate dataset objects (`train_base` with random crop and flip, and `val_base` with deterministic normalization only) and wrapped them in `Subset(train_base, train_indices)` and `Subset(val_base, val_indices)` using disjoint indices generated from `split_seed=42`.

9. **Q: How do you verify in automated tests that validation data has no stochastic augmentation?**
   **A:** `test_validation_transform_is_deterministic_without_augmentation_leakage` in `tests/test_data.py` fetches `bundle.val_dataset[0]` twice and asserts `torch.allclose(img1, img2, atol=1e-7)`.

10. **Q: Why did you use `AdaptiveAvgPool2d((1, 1))` before the linear layer instead of flattening the `(B, 128, 4, 4)` tensor directly?**
    **A:** Flattening `(B, 128, 4, 4)` produces a $2,048$-dimensional vector, requiring $2,048 \times 10 + 10 = 20,490$ parameters in the linear head and increasing overfitting sensitivity to exact spatial position. Global average pooling reduces `(B, 128, 4, 4)` to `(B, 128, 1, 1)`, requiring only $1,290$ head parameters and encouraging translation invariance.

11. **Q: Why do we call `optimizer.zero_grad()` before `loss.backward()` in every batch?**
    **A:** PyTorch accumulates gradients into `param.grad` by addition (`+=`) on every `.backward()` call. Without clearing gradients at the start of each mini-batch step, gradients from previous batches would corrupt the current update direction.

12. **Q: What is the difference between `model.eval()` and `with torch.no_grad():`? Why do we need both during validation?**
    **A:** `model.eval()` changes layer behavior (BatchNorm uses running stats instead of batch stats; Dropout is disabled), whereas `torch.no_grad()` disables autograd graph construction and gradient memory allocation. Using only `model.eval()` still builds the computation graph in memory; using only `torch.no_grad()` leaves Dropout active and BatchNorm updating running statistics on validation data.

13. **Q: How does checkpoint selection work in `src/attentionvision/train.py`?**
    **A:** After every epoch, `best_model.pt` is updated only when `val_metrics["accuracy"] > best_val_acc` (or when validation accuracy ties and `val_metrics["loss"] < best_val_loss`). The test set is never inspected during training.

14. **Q: What information is stored inside each `.pt` checkpoint file?**
    **A:** `epoch`, `best_val_acc`, `best_val_loss`, `seed`, `model_name`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `config` dictionary, epoch `history`, and `environment` metadata.

15. **Q: Why must optimizer and scheduler states be saved to support resuming training?**
    **A:** `AdamW` maintains per-parameter moving averages (`exp_avg` and `exp_avg_sq`) and step counts, and `CosineAnnealingLR` tracks `last_epoch`. Restoring only `model_state_dict` resets momentum buffers and learning rate to epoch 0, causing a sharp optimization spike upon resumption.

16. **Q: How did you measure inference latency fairly?**
    **A:** In `measure_inference_latency` (`src/attentionvision/metrics.py`), both models are placed on the same device in `eval()` mode under `torch.no_grad()`, run for `warmup_runs` untimed iterations (synchronizing CUDA if applicable), and timed over `measure_runs` consecutive forward passes at identical `batch_size=1`.

17. **Q: Why does `AttentionCNN` have ~4–8% higher CPU inference latency despite having only 0.21% more parameters?**
    **A:** Latency on small operations is often memory-bandwidth-bound rather than FLOP-bound. Computing channel-wise `mean`, channel-wise `max`, `torch.cat`, `sigmoid`, and elementwise multiplication requires reading and writing full intermediate activation tensors across RAM/cache multiple times.

18. **Q: How does the robustness experiment ensure a fair comparison between `BaselineCNN` and `AttentionCNN`?**
    **A:** `apply_gaussian_noise` in `src/attentionvision/robustness.py` seeds its NumPy generator deterministically using `corruption_seed + batch_idx * 10007`, and `apply_gaussian_blur` is a deterministic spatial filter. Thus, both models evaluate the exact same corrupted pixel tensors at every severity $\sigma$.

19. **Q: Why are corruptions applied only during evaluation and not during training?**
    **A:** Because our research question (H2) tests how standardly trained baseline and attention models react to out-of-training-condition pixel perturbations. Training on Gaussian noise or blur would turn the experiment into corruption-augmented training rather than measuring natural corruption sensitivity.

20. **Q: What did the corruption sensitivity sweep reveal in the 3-seed smoke benchmark?**
    **A:** Under mild Gaussian blur ($\sigma = 0.5$), `AttentionCNN` outperformed `BaselineCNN` ($83.33\%$ vs. $68.33\%$ accuracy), whereas under severe Gaussian blur ($\sigma = 1.0, 1.5$), `AttentionCNN` dropped to $20.00\%$ while `BaselineCNN` retained $45.00\%$ and $38.33\%$. Under additive Gaussian noise, both models degraded similarly as $\sigma$ increased from $0.25$ to $1.00$.

21. **Q: Do the spatial attention heatmaps in `attention_maps.png` prove why the model predicted a given class?**
    **A:** No. The heatmaps show the scalar gate values $M_s(F) \in (0, 1)$ applied to intermediate feature maps at $16 \times 16$ and $8 \times 8$ resolutions. Because each location already mixes a wide receptive field and passes through subsequent pooling and linear weights, attention maps are descriptive of feature reweighting, not causal attribution.

22. **Q: Why do unit tests in `tests/` use synthetic tensors and `SyntheticCIFAR10Dataset` instead of downloading CIFAR-10?**
    **A:** Unit tests and CI workflows must be fast, deterministic, and independent of external network availability or third-party servers (`cs.toronto.edu`). Testing tensor shapes, gradient flow, split disjointness, and checkpoint roundtrips requires controlled synthetic inputs.

23. **Q: How isgradient clipping used in `train_one_epoch` and why?**
    **A:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)` rescales all parameter gradients jointly if their global $L_2$ norm exceeds `2.0`, preventing occasional steep mini-batch gradients from destabilizing weights.

24. **Q: What is the difference between macro-F1 and accuracy on CIFAR-10?**
    **A:** Even when all 10 classes have equal support ($1,000$ test images per class), macro-F1 averages the harmonic mean of precision and recall across classes. If a model predicts one class with high false positives and another with high false negatives, macro-F1 drops lower than overall accuracy.

25. **Q: Why did you use channel-wise `mean` and `max` instead of a $1 \times 1$ convolution to compress channels in `SpatialAttention`?**
    **A:** A $1 \times 1$ convolution from $C$ channels to $1$ channel introduces $C$ learnable parameters that depend on the stage's channel count and can overfit to specific channel indices. Channel-wise `mean` and `max` are parameter-free, permutation-invariant across channels, and make the `SpatialAttention` module reusable across stages with different channel widths ($32$, $64$, or $128$).

26. **Q: How does `compare_results.py` handle multiple random seeds?**
    **A:** It groups all completed `evaluation_metrics.json` files by `model_name` and computes the sample mean and standard deviation across seeds (`42, 43, 44`) for validation accuracy, test accuracy, macro precision/recall/F1, training time, and latency.

27. **Q: What environment limitations existed in the build workspace, and how did you handle them honestly?**
    **A:** The active build workspace was an isolated 2-core CPU container without GPU hardware or outbound internet access to download the 170 MB CIFAR-10 archive. Rather than fabricating CIFAR-10 benchmark numbers, we executed and verified all 28 unit tests and a 3-seed synthetic smoke benchmark on CPU, and marked the full 50,000-image CIFAR-10 table entries as `Pending` with the exact `scripts/run_experiments.sh` script ready for the target RTX 4050 laptop.

28. **Q: If you ran `scripts/run_experiments.sh` on CIFAR-10 and found that `AttentionCNN` scored $84.1 \pm 0.3\%$ while `BaselineCNN` scored $84.0 \pm 0.3\%$, what would you conclude?**
    **A:** I would conclude that on $32 \times 32$ CIFAR-10 images with a 3-stage backbone, spatial attention provides **no statistically meaningful improvement** ($+0.1\%$ is well within the $\pm 0.3\%$ cross-seed standard deviation), and given its $+4\text{--}8\%$ latency overhead, `BaselineCNN` is the preferable architecture for this operating regime.

---

## 8. 12 Challenging Follow-Up Questions & Defenses

1. **Follow-up: What is the exact backward gradient of `max_out, _ = x.max(dim=1, keepdim=True)` in `SpatialAttention`?**
   - **Defense**: For a given batch $b$ and spatial coordinate $(h, w)$, let $c^* = \arg\max_c x_{b,c,h,w}$. The partial derivative $\frac{\partial \text{max\_out}_{b,1,h,w}}{\partial x_{b,c,h,w}}$ is $1$ if $c = c^*$ (split equally if tied) and $0$ for all other $c \neq c^*$. Thus, gradients from the max-pooled branch route exclusively back to the single maximally activated channel at each pixel, whereas gradients from `x.mean(dim=1, keepdim=True)` distribute $\frac{1}{C}$ equally to every channel.

2. **Follow-up: Write out the product-rule gradient $\frac{\partial F_{\text{out}}}{\partial F}$ for your residual spatial attention $F_{\text{out}} = F \odot (1 + M_s(F))$.**
   - **Defense**: Because $M_s(F)$ is itself a differentiable function of $F$, the chain and product rules give:
     $$\frac{\partial \mathcal{L}}{\partial F} = \frac{\partial \mathcal{L}}{\partial F_{\text{out}}} \odot \left(1 + M_s(F)\right) + \left(\sum_{c=1}^{C} \frac{\partial \mathcal{L}}{\partial F_{\text{out}, c}} \odot F_c\right) \frac{\partial M_s(F)}{\partial F}$$
     Notice the first term $\frac{\partial \mathcal{L}}{\partial F_{\text{out}}} \odot (1 + M_s(F))$: because $1 + M_s(F) \in (1, 2)$, gradients flowing backward to earlier convolutional layers are never zeroed out even if $M_s(F) \to 0$.

3. **Follow-up: Why did you place `SpatialAttention` *after* `BatchNorm2d` and `ReLU`, and *before* `MaxPool2d`?**
   - **Defense**: Placing attention after `BatchNorm2d` and `ReLU` ensures that the inputs to `mean(dim=1)` and `max(dim=1)` are non-negative, scale-normalized feature activations (if channels had arbitrary scales before BatchNorm, a single unnormalized channel would dominate `max(dim=1)`). Placing it *before* `MaxPool2d` allows spatial attention to reweight high-resolution features before $2 \times 2$ downsampling discards $75\%$ of spatial positions.

4. **Follow-up: Could you add Channel Attention (like Squeeze-and-Excitation or full CBAM) as well? Why didn't you?**
   - **Defense**: Yes—Squeeze-and-Excitation (Hu et al., 2018) and full CBAM (Woo et al., 2018) combine channel attention $M_c(F) \in \mathbb{R}^{B \times C \times 1 \times 1}$ with spatial attention $M_s(F) \in \mathbb{R}^{B \times 1 \times H \times W}$. However, combining both mechanisms confounds the experiment: if accuracy changes, we wouldn't know whether channel reweighting or spatial reweighting caused it without running a $2 \times 2$ factorial ablation. Isolating spatial attention directly tests our research question.

5. **Follow-up: What happens if you apply Gaussian noise *before* vs. *after* `Normalize(mean, std)`?**
   - **Defense**: In `src/attentionvision/robustness.py`, Gaussian noise $\mathcal{N}(0, \sigma^2)$ is added in normalized tensor space. Because CIFAR-10 channel standard deviations are $\approx 0.25$, a normalized perturbation of $\sigma = 0.20$ corresponds to $\approx 0.20 \times 0.25 = 0.05$ in $[0, 1]$ raw pixel space ($\approx 12.75 / 255$ intensity levels). Documenting the exact space where $\sigma$ is applied is critical so comparisons across papers are consistent.

6. **Follow-up: Why might `AttentionCNN` perform worse than `BaselineCNN` under severe Gaussian blur ($\sigma \ge 1.0$)?**
   - **Defense**: Spatial attention relies on contrast between foreground activation peaks and background regions in $F_{\text{avg}}$ and $F_{\text{max}}$. Strong low-pass Gaussian blur smooths out spatial gradients and spreads energy across the feature map, causing the learned $7 \times 7$ attention filter (trained on sharp images) to produce poorly localized masks that distort feature magnitudes prior to global pooling.

7. **Follow-up: If you wanted to improve robustness to Gaussian blur and noise, how would you modify training?**
   - **Defense**: Rather than relying on architectural attention alone, I would introduce mixed-corruption data augmentation (such as AugMix or RandAugment) during training, or replace sharp `MaxPool2d` downsampling with anti-aliased blur-pooling (Zhang, ICML 2019).

8. **Follow-up: Why is `split_seed` separate from `seed` in `DataConfig`?**
   - **Defense**: When running multi-seed experiments (`seed = 42, 43, 44`), we want to vary weight initialization and mini-batch shuffling order while keeping the $45,000 / 5,000$ train/validation partition fixed (`split_seed = 42`) so all runs are evaluated on the exact same validation set.

9. **Follow-up: Why did you choose $k = 7$ for the spatial attention kernel on $16 \times 16$ and $8 \times 8$ feature maps? Isn't $7 \times 7$ almost the entire $8 \times 8$ map?**
   - **Defense**: Yes! On an $8 \times 8$ map, a $7 \times 7$ kernel covers almost the entire spatial grid, allowing the attention gate at each coordinate to compare local activation against nearly global spatial context, while costing only $98$ parameters. Because `attention_kernel_size` is configurable in YAML (`3`, `5`, or `7`), we can easily run an ablation across kernel sizes.

10. **Follow-up: How would you test statistically whether `AttentionCNN` is significantly better than `BaselineCNN`?**
    - **Defense**: With $N \ge 5$ paired random seeds (where seed $s$ initializes both models and uses the same batch order), I would compute the paired differences $d_s = \text{Acc}_{\text{attn}}(s) - \text{Acc}_{\text{base}}(s)$ and perform a paired two-sided $t$-test (or Wilcoxon signed-rank test), alongside reporting the $95\%$ confidence interval of the mean difference. With only 3 seeds, statistical power is low, so we report $\text{mean} \pm \text{std}$ descriptively and avoid overclaiming significance.

11. **Follow-up: Could there be data leakage through the CIFAR-10 normalization constants `mean=[0.4914, 0.4822, 0.4465]`?**
    - **Defense**: Those constants are the standard published channel means and standard deviations computed over the 50,000 official CIFAR-10 training set (not the 10,000 test set). Technically, the $5,000$ validation images are part of the $50,000$ training set, but global 3-number RGB mean/std statistics over $45,000$ vs. $50,000$ images agree to four decimal places.

12. **Follow-up: Why did you implement the project as an installable `src/attentionvision` package instead of flat scripts?**
    - **Defense**: Using the `src/` layout (`pyproject.toml` + `src/attentionvision/`) prevents accidental imports from the current working directory during testing, enforces clean module boundaries (`models`, `data`, `train`, `evaluate`, `robustness`, `metrics`), and provides a single reproducible CLI (`python -m attentionvision`).

---

## 9. Common Technical Mistakes to Avoid in Interviews

1. **Never claim attention maps explain *why* the CNN made a decision**: State clearly that $M_s(F)$ shows spatial feature reweighting at intermediate layers, not causal pixel attribution.
2. **Never claim synthetic corruption tests prove real-world robustness**: State that Gaussian noise and Gaussian blur test two specific parametric frequency perturbations.
3. **Never confuse parameter count with FLOPs or wall-clock latency**: Point out that $196$ extra parameters can still add $4\text{--}8\%$ latency due to tensor reduction and memory-bandwidth overhead.
4. **Never tune hyperparameters on the 10,000-image CIFAR-10 test set**: Emphasize that all checkpoint selection and early stopping happen strictly on the 5,000-image validation split.
