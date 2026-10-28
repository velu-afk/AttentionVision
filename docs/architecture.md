# Model and Spatial Attention Architecture

## 1. Architectural Design Philosophy

AttentionVision compares two compact convolutional neural networks designed for $32 \times 32$ RGB images:
1. **`BaselineCNN` (Model A)**: A 3-stage convolutional backbone with batch normalization, ReLU nonlinearities, $2 \times 2$ max pooling, global average pooling, dropout, and a linear classifier.
2. **`AttentionCNN` (Model B)**: The exact same 3-stage backbone augmented with a lightweight channel-pooled **Spatial Attention** module (`SpatialAttention`) at intermediate stages (`Stage 2` at $16 \times 16$ resolution and `Stage 3` at $8 \times 8$ resolution).

Using global average pooling (`AdaptiveAvgPool2d((1, 1))`) instead of flattening the $4 \times 4 \times 128 = 2048$-dimensional feature map keeps the classification head compact ($128 \times 10 + 10 = 1,290$ parameters), reduces overfitting on small images, and keeps total model size under $95,000$ parameters for fast laptop training.

## 2. Stage-by-Stage Tensor Dimensions

For an input mini-batch of $B$ RGB images of shape $(B, 3, 32, 32)$:

| Stage / Layer | Operation | Input Shape | Output Shape | Parameters (`BaselineCNN`) | Parameters (`AttentionCNN`) |
| :--- | :--- | :--- | :--- | ---: | ---: |
| **Stage 1 (`block1`)** | `Conv2d(3, 32, k=3, s=1, p=1, bias=False)` | $(B, 3, 32, 32)$ | $(B, 32, 32, 32)$ | $864$ | $864$ |
| | `BatchNorm2d(32)` + `ReLU` | $(B, 32, 32, 32)$ | $(B, 32, 32, 32)$ | $64$ | $64$ |
| **Pool 1 (`pool1`)** | `MaxPool2d(k=2, s=2)` | $(B, 32, 32, 32)$ | $(B, 32, 16, 16)$ | $0$ | $0$ |
| **Stage 2 (`block2`)** | `Conv2d(32, 64, k=3, s=1, p=1, bias=False)` | $(B, 32, 16, 16)$ | $(B, 64, 16, 16)$ | $18,432$ | $18,432$ |
| | `BatchNorm2d(64)` + `ReLU` | $(B, 64, 16, 16)$ | $(B, 64, 16, 16)$ | $128$ | $128$ |
| **Attention 2 (`attn2`)** | `SpatialAttention(k=7, residual=True)` | $(B, 64, 16, 16)$ | $(B, 64, 16, 16)$ | — | $98$ |
| **Pool 2 (`pool2`)** | `MaxPool2d(k=2, s=2)` | $(B, 64, 16, 16)$ | $(B, 64, 8, 8)$ | $0$ | $0$ |
| **Stage 3 (`block3`)** | `Conv2d(64, 128, k=3, s=1, p=1, bias=False)` | $(B, 64, 8, 8)$ | $(B, 128, 8, 8)$ | $73,728$ | $73,728$ |
| | `BatchNorm2d(128)` + `ReLU` | $(B, 128, 8, 8)$ | $(B, 128, 8, 8)$ | $256$ | $256$ |
| **Attention 3 (`attn3`)** | `SpatialAttention(k=7, residual=True)` | $(B, 128, 8, 8)$ | $(B, 128, 8, 8)$ | — | $98$ |
| **Pool 3 (`pool3`)** | `MaxPool2d(k=2, s=2)` | $(B, 128, 8, 8)$ | $(B, 128, 4, 4)$ | $0$ | $0$ |
| **Global Pool** | `AdaptiveAvgPool2d((1, 1))` + `Flatten` | $(B, 128, 4, 4)$ | $(B, 128)$ | $0$ | $0$ |
| **Classifier Head** | `Dropout(p=0.25)` + `Linear(128, 10)` | $(B, 128)$ | $(B, 10)$ | $1,290$ | $1,290$ |
| **Total Trainable** | | | | **$94,762$** | **$94,958$ ($+196$)** |

Why is `bias=False` used in `Conv2d` layers? Because `BatchNorm2d` immediately subtracts the per-channel spatial mean $\mu_c$ and adds its own learnable shift parameter $\beta_c$, making a preceding convolutional bias redundant.

## 3. Spatial Attention Module (`src/attentionvision/models/attention.py`)

### 3.1 Mathematical Formulation
Given an intermediate feature tensor $F \in \mathbb{R}^{B \times C \times H \times W}$:

1. **Channel-Wise Aggregation**:
   To summarize spatial activity across all $C$ channels without introducing channel-dependent parameters, we compute both the channel-wise mean and channel-wise maximum at every spatial coordinate $(h, w)$:
   $$F_{\text{avg}}(b, 1, h, w) = \frac{1}{C} \sum_{c=1}^{C} F(b, c, h, w) \in \mathbb{R}^{B \times 1 \times H \times W}$$
   $$F_{\text{max}}(b, 1, h, w) = \max_{c \in \{1, \dots, C\}} F(b, c, h, w) \in \mathbb{R}^{B \times 1 \times H \times W}$$
2. **Spatial Descriptor Concatenation**:
   $$F_{\text{cat}} = \text{Concat}([F_{\text{avg}}, F_{\text{max}}], \text{dim}=1) \in \mathbb{R}^{B \times 2 \times H \times W}$$
3. **Spatial Context Convolution and Sigmoid Gating**:
   A single 2D convolutional layer with $C_{\text{in}} = 2$, $C_{\text{out}} = 1$, odd kernel size $k \times k$ (default $k = 7$), stride $1$, and symmetric padding $p = (k - 1) / 2$ maps the 2-channel descriptor to a 1-channel logit map, followed by the logistic sigmoid $\sigma(z) = (1 + e^{-z})^{-1}$:
   $$M_s(F) = \sigma\left(\text{Conv2d}_{2 \to 1, k \times k}(F_{\text{cat}})\right) \in (0, 1)^{B \times 1 \times H \times W}$$
4. **Broadcasted Feature Modulation**:
   In PyTorch, multiplying a tensor of shape $(B, C, H, W)$ by a tensor of shape $(B, 1, H, W)$ automatically broadcasts the singleton channel axis across all $C$ channels:
   - **Residual Multiplicative Gating (`residual=True`, default)**:
     $$F_{\text{out}} = F \odot \left(1 + M_s(F)\right) \in \mathbb{R}^{B \times C \times H \times W}$$
   - **Direct Multiplicative Gating (`residual=False`)**:
     $$F_{\text{out}} = F \odot M_s(F) \in \mathbb{R}^{B \times C \times H \times W}$$

### 3.2 How Spatial Attention Differs from Standard Convolution
- A standard `Conv2d` layer applies **input-independent** (static) spatial filters $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$ uniformly across all spatial positions via sliding dot products.
- `SpatialAttention` computes an **input-dependent** (dynamic) spatial mask $M_s(F)$ for each individual image in the batch and scales feature activations multiplicatively ($F \odot (1 + M_s(F))$).

### 3.3 Parameter and Computational Cost
- **Parameters**: Each `SpatialAttention(kernel_size=k, bias=False)` layer contains exactly $2 \times 1 \times k \times k = 2k^2$ parameters ($98$ parameters for $k=7$). Across Stages 2 and 3, this adds $196$ parameters ($+0.21\%$ over `BaselineCNN`).
- **FLOPs (Multiply-Adds)**: At spatial resolution $H \times W$, channel pooling requires $2 C H W$ operations, the $2 \to 1$ convolution requires $2 k^2 H W$ multiply-adds, and modulation requires $C H W$ multiplications. At Stage 2 ($64 \times 16 \times 16$) and Stage 3 ($128 \times 8 \times 8$), this adds roughly $6.1 \times 10^4$ operations per image—under $0.5\%$ of the backbone convolution FLOPs, though extra kernel launches and memory reads add ~3–8% wall-clock latency on CPU.

### 3.4 Why Attention Might Help vs. Why It Might Fail
- **Why it might help**: Before $2 \times 2$ max pooling and final global average pooling, background clutter can contribute noisy activations to the pooled representation. Spatial attention can amplify foreground object regions relative to uninformative background tiles.
- **Why it might fail**: CIFAR-10 images are tiny ($32 \times 32$). By Stage 3, feature maps are $8 \times 8$, and each $3 \times 3$ backbone filter already has a large effective receptive field. Furthermore, if strong image corruption (such as heavy Gaussian blur) washes out high-frequency edges, the channel-wise max/mean statistics can mislocalize, causing the attention mask to suppress useful residual cues.
- **Interpretability Caveat**: Attention maps $M_s(F)$ visualize where the module scales intermediate activations; they are **not** causal explanations or saliency proofs of the final class decision.
