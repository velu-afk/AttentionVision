# Methodological and Experimental Limitations

A strong machine learning engineer understands not only what an experiment shows, but also what it **cannot** prove. Below are the core limitations of AttentionVision that you should discuss candidly in technical reviews and interviews.

## 1. Small Spatial Resolution of CIFAR-10 ($32 \times 32$)

CIFAR-10 images are only $32 \times 32$ pixels. After two $2 \times 2$ max-pooling layers, the feature maps entering Stage 3 have spatial dimensions of just $8 \times 8$.
- At $32 \times 32$ and $16 \times 16$ resolutions, objects often occupy most of the frame, leaving limited background clutter for a spatial-attention mask to suppress.
- Spatial attention mechanisms typically exhibit larger gains on higher-resolution benchmarks (such as ImageNet at $224 \times 224$ or object detection datasets) where target objects occupy a small fraction of a cluttered scene.

## 2. Single-Dataset Scope

All experiments in this repository are conducted on CIFAR-10 (or its procedural 32x32 synthetic counterpart during offline smoke tests). Findings on CIFAR-10 do not automatically generalize to:
- Fine-grained classification datasets (e.g., Oxford Pets, CUB-200).
- Medical imaging or satellite imagery with different spectral and spatial statistics.
- High-resolution multi-object scenes.

## 3. Compact Backbone Capacity

To ensure fast, accessible training on a single laptop (or CPU), `BaselineCNN` and `AttentionCNN` use a 3-stage backbone with $94,762$ and $94,958$ parameters, respectively.
- Deeper residual architectures (such as ResNet-18 or ResNet-50 with $>11\text{M}$ parameters) have different gradient-flow dynamics and feature hierarchies, so the marginal effect of adding CBAM-style spatial attention may differ in deep residual networks compared with a shallow 3-stage CNN.

## 4. Non-Causal Nature of Attention Visualizations

While `plot_attention_maps` in `src/attentionvision/evaluate.py` visualizes the $16 \times 16$ and $8 \times 8$ spatial attention maps $M_s(F) \in (0, 1)^{B \times 1 \times H \times W}$, these heatmaps must **not** be interpreted as causal explanations of model predictions:
- By Stage 2 and Stage 3, each spatial coordinate $(h, w)$ has an effective receptive field spanning a $7 \times 7$ to $15 \times 15$ patch of the input image, mixing information from neighboring pixels.
- Multiplying a feature map by $1 + M_s(F)$ reweights intermediate activations prior to global average pooling; high attention weight at coordinate $(h, w)$ does not prove that the corresponding input pixels caused the predicted class label (see Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).

## 5. Synthetic Corruption Scope vs. Real-World Robustness

The robustness module (`src/attentionvision/robustness.py`) applies additive Gaussian noise and isotropic Gaussian blur in tensor space during evaluation.
- These two parametric perturbations test high-frequency noise sensitivity and low-pass edge degradation, respectively.
- They do **not** capture real-world distribution shifts such as lighting changes, camera sensor saturation, motion blur, pose variation, background domain shift, or adversarial perturbations ($L_\infty$ / $L_2$ PGD attacks).

## 6. Cross-Hardware Reproducibility Boundaries

Although `set_seed` fixes Python, NumPy, and PyTorch random number generators and enables deterministic cuDNN algorithms, bitwise identical floating-point results are guaranteed only across runs on the **same hardware and software stack**. Running the same seed on an NVIDIA RTX 4050 GPU versus an x86_64 CPU will produce slightly different floating-point trajectories due to parallel atomic reduction order.
