# Machine Learning & Deep Learning Interview Guide (Beginner to Interview-Ready)

This guide teaches every core concept used in **AttentionVision** from first principles so you can explain both the mathematical intuition and the exact PyTorch implementation during an ML internship interview.

---

## 1. Supervised Learning
- **Intuition**: Learning a mapping $f_\theta: \mathcal{X} \to \mathcal{Y}$ from labeled input-output pairs $\{(x_i, y_i)\}_{i=1}^{N}$ so that $f_\theta$ generalizes to unseen inputs drawn from the same distribution.
- **In AttentionVision**: Each input $x_i \in \mathbb{R}^{3 \times 32 \times 32}$ is a normalized RGB image, and each label $y_i \in \{0, 1, \dots, 9\}$ is an integer representing one of the 10 CIFAR-10 object categories.
- **Common Follow-up**: *Why can't we just measure success on the training set?* Because a high-capacity network can memorize training examples (achieving near-zero training loss) while failing on new images.

## 2. Neural-Network Fundamentals
- **Intuition**: A neural network composes simple linear transformations ($z = W x + b$) and elementwise nonlinear activations ($\sigma(z)$) across multiple layers:
  $$h^{(l)} = \sigma\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)$$
- **Why Nonlinearity Matters (`ReLU(z) = max(0, z)`)**: Without nonlinear activations between layers, a stack of linear layers collapses into a single linear matrix multiplication ($W^{(2)} (W^{(1)} x) = (W^{(2)} W^{(1)}) x$) and cannot represent curved decision boundaries.

## 3. Forward and Backward Propagation
- **Forward Pass**: Data flows from input tensor $x$ through convolutional blocks, optional spatial attention, pooling, and the linear head to produce logits $\hat{z} \in \mathbb{R}^{B \times 10}$ and scalar loss $\mathcal{L}$.
- **Backward Pass (Backpropagation)**: Applying the multivariable chain rule from $\mathcal{L}$ backward through the computation graph to compute $\frac{\partial \mathcal{L}}{\partial \theta}$ for every learnable parameter $\theta$.
- **In Code (`src/attentionvision/train.py`)**:
  - `logits = model(images)` executes the forward pass.
  - `loss.backward()` traverses the autograd graph backward and populates `param.grad`.

## 4. Loss Functions (Cross-Entropy Loss)
- **Intuition**: Cross-entropy measures the Kullback-Leibler divergence between the true one-hot class distribution and the model's predicted softmax probability distribution.
- **Equation**: Given raw class logits $z = (z_0, \dots, z_{C-1}) \in \mathbb{R}^C$ and ground-truth class index $y \in \{0, \dots, C-1\}$:
  $$p_c = \text{Softmax}(z)_c = \frac{e^{z_c}}{\sum_{j=0}^{C-1} e^{z_j}}, \quad \mathcal{L}_{\text{CE}}(z, y) = -\log p_y = -z_y + \log \sum_{j=0}^{C-1} e^{z_j}$$
- **Simple Example**: If the true class is $y = 2$ and the model predicts probability $p_2 = 0.90$, then $\mathcal{L} = -\ln(0.90) \approx 0.105$. If the model is confident but wrong ($p_2 = 0.01$), $\mathcal{L} = -\ln(0.01) \approx 4.605$.
- **Common Follow-up**: *Why does `nn.CrossEntropyLoss` take raw logits instead of softmax probabilities?* Combining `LogSoftmax` and `NLLLoss` uses the log-sum-exp trick ($\log \sum_j e^{z_j - \max_k z_k}$), preventing numerical overflow/underflow when logits are large.

## 5. Gradient Descent and Mini-Batch SGD / AdamW
- **Intuition**: The gradient $\nabla_\theta \mathcal{L}$ points in the direction of steepest loss increase. Stepping in the opposite direction reduces loss:
  $$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$
- **AdamW (`torch.optim.AdamW`)**: Maintains exponential moving averages of the first moment (mean gradient $m_t$) and second moment (uncentered variance $v_t$) to adapt step sizes per parameter, while applying **decoupled weight decay** ($\theta \leftarrow (1 - \eta \lambda)\theta$) directly to parameters rather than mixing $L_2$ regularization into the adaptive gradient denominator.

## 6. Learning Rate and Cosine Annealing
- **Intuition**: The learning rate $\eta$ controls step size. Too large $\to$ divergence/oscillation; too small $\to$ slow convergence or getting trapped in sharp local regions.
- **Cosine Annealing (`CosineAnnealingLR`)**: Smoothly decays $\eta_t$ from $\eta_{\max}$ to $\eta_{\min}$ over $T_{\max}$ epochs:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T_{\max}}\right)\right)$$

## 7. Convolutions (`nn.Conv2d`)
- **Intuition**: Instead of connecting every input pixel to every hidden neuron (which would require $3072 \times H$ parameters in the very first layer and destroy 2D spatial topology), a 2D convolution slides small learnable filters across height and width.
- **Two Key Properties**:
  1. **Local Connectivity**: Each output activation depends only on a small local patch (e.g., $3 \times 3$).
  2. **Parameter Sharing (Translation Equivariance)**: The same filter weights are applied at every $(h, w)$ position; if an edge or wheel shifts in the image, its feature activation shifts by the same amount.

## 8. Kernels, Strides, and Padding
- **Spatial Output Dimension Formula**: For input height/width $H_{\text{in}}$, kernel size $k$, padding $p$, and stride $s$:
  $$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$
- **Concrete Examples in AttentionVision**:
  - **Backbone `Conv2d` ($k=3, p=1, s=1$)**: $H_{\text{out}} = \lfloor(32 + 2(1) - 3)/1\rfloor + 1 = 32$ (preserves spatial resolution).
  - **SpatialAttention `Conv2d` ($k=7, p=3, s=1$)**: $H_{\text{out}} = \lfloor(H + 2(3) - 7)/1\rfloor + 1 = H$ (preserves spatial resolution).
  - **`MaxPool2d` ($k=2, p=0, s=2$)**: $H_{\text{out}} = \lfloor(32 + 0 - 2)/2\rfloor + 1 = 16$ (halves spatial resolution).

## 9. Feature Maps
- **Definition**: A 4D tensor $F \in \mathbb{R}^{B \times C \times H \times W}$ where $B$ is batch size, $C$ is the number of feature channels (each corresponding to one learned convolutional filter), and $H \times W$ is the spatial grid.
- **Hierarchy**: Early layers ($C=32, 32 \times 32$) detect edges and color contrasts; middle layers ($C=64, 16 \times 16$) detect textures and corners; deeper layers ($C=128, 8 \times 8$) detect object parts.

## 10. Receptive Fields
- **Intuition**: The region of the original $32 \times 32$ input image that can influence a single activation at layer $l$.
- **Recurrence**: $R_l = R_{l-1} + (k_l - 1) \prod_{i=1}^{l-1} s_i$.
- **In AttentionVision**:
  - After `block1` ($3 \times 3, s=1$): $R = 3 \times 3$.
  - After `pool1` ($2 \times 2, s=2$): $R = 4 \times 4$ (stride becomes $2$).
  - After `block2` ($3 \times 3, s=1$): $R = 4 + (3 - 1) \times 2 = 8 \times 8$.
  - After `pool2` ($2 \times 2, s=2$): $R = 8 + (2 - 1) \times 2 = 10 \times 10$ (cumulative stride becomes $4$).
  - After `block3` ($3 \times 3, s=1$): $R = 10 + (3 - 1) \times 4 = 18 \times 18$ pixels of the $32 \times 32$ input!

## 11. Pooling (`MaxPool2d` and `AdaptiveAvgPool2d`)
- **`MaxPool2d(2, 2)`**: Takes the maximum value in each $2 \times 2$ window, reducing spatial dimensions by $2\times$ ($4\times$ fewer spatial elements) and providing local translation invariance.
- **`AdaptiveAvgPool2d((1, 1))`**: Averages each of the $128$ channels across its $4 \times 4$ spatial grid into a single scalar per channel, producing a $(B, 128, 1, 1)$ tensor without adding any learnable parameters.

## 12. Batch Normalization (`nn.BatchNorm2d`)
- **Equation**: For channel $c$, computes mini-batch spatial mean $\mu_c$ and variance $\sigma_c^2$ over $(B, H, W)$:
  $$\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \quad y_{b,c,h,w} = \gamma_c \hat{x}_{b,c,h,w} + \beta_c$$
- **Train vs. Eval Behavior**: During `model.train()`, BatchNorm normalizes using current batch statistics and updates exponential moving averages `running_mean` and `running_var`. During `model.eval()`, it freezes and uses `running_mean` and `running_var` so predictions on a single test image are deterministic and independent of other images in the batch.

## 13. Dropout (`nn.Dropout`)
- **Intuition**: During training (`model.train()`), randomly zeroes each feature activation with probability $p$ (default $0.25$) and scales remaining activations by $\frac{1}{1 - p}$ (inverted dropout) so expected activation magnitude is unchanged during `model.eval()` when dropout is disabled. Prevents co-adaptation of features.

## 14. Data Augmentation
- **In AttentionVision**: `RandomCrop(32, padding=4)` and `RandomHorizontalFlip(p=0.5)` applied **only** to the training split. Teach the network invariance to small horizontal translations and left-right orientation without collecting extra labeled images.

## 15. Overfitting and Generalization Gap
- **Diagnosis**: Training loss continues to drop toward $0$ (and training accuracy approaches $100\%$) while validation loss stops decreasing and begins to rise.
- **Mitigations Used in AttentionVision**: Data augmentation, BatchNorm, Global Average Pooling (instead of a large dense MLP), Dropout ($p=0.25$), decoupled weight decay ($\lambda = 5 \times 10^{-4}$), and early stopping.

## 16. Train, Validation, and Test Splits
- **Training Set ($45,000$ images)**: Used to compute gradients and update weights $\theta$.
- **Validation Set ($5,000$ images)**: Used after each epoch to monitor generalization, trigger early stopping, and select `best_model.pt`.
- **Test Set ($10,000$ images)**: Held out until training is complete to obtain an unbiased estimate of final performance.

## 17. Accuracy, Precision, Recall, and Macro-F1
- **Accuracy**: Overall fraction of correct predictions. On balanced 10-class CIFAR-10 ($1,000$ test images per class), accuracy equals macro recall.
- **Precision ($P_c$)**: Of all images predicted as class $c$, what fraction truly belong to class $c$?
- **Recall ($R_c$)**: Of all images that truly belong to class $c$, what fraction did the model detect?
- **Macro-F1**: $\frac{1}{C} \sum_{c=0}^{C-1} \frac{2 P_c R_c}{P_c + R_c}$. Penalizes models that perform poorly on specific classes even if overall accuracy looks acceptable.

## 18. Confusion Matrices
- **Definition**: A $C \times C$ table where row $i$ is the true class and column $j$ is the predicted class. Diagonal entries $M_{i,i}$ are correct predictions; off-diagonal entries $M_{i,j}$ ($i \neq j$) reveal systematic class confusions (e.g., `cat` vs. `dog`, or `automobile` vs. `truck`).

## 19. Spatial Attention and Attention Maps
- **Core Idea**: Not all spatial pixels in a feature map are equally informative for classification. Spatial attention learns a function $M_s: \mathbb{R}^{B \times C \times H \times W} \to (0, 1)^{B \times 1 \times H \times W}$ that assigns a weight between $0$ and $1$ to each $(h, w)$ location:
  $$M_s(F) = \sigma\left(\text{Conv2d}_{2 \to 1, k \times k}\left([\text{Mean}_c(F); \text{Max}_c(F)]\right)\right)$$
- **Residual Modulation**: $F_{\text{out}} = F \odot (1 + M_s(F))$.

## 20. Computational Complexity and Parameter vs. Latency Trade-offs
- **Why Parameter Count $\neq$ Inference Latency**: Adding two $7 \times 7$ spatial-attention modules adds only $196$ parameters ($0.21\%$), because the $2 \to 1$ convolution shares its $98$ weights across every spatial location. However, computing `mean(dim=1)`, `max(dim=1)`, `cat`, `conv2d`, `sigmoid`, and elementwise multiplication reads and writes the full $(B, C, H, W)$ activation tensor in memory, adding ~4–8% wall-clock latency on CPU/GPU due to memory-bandwidth overhead.

## 21. Ablation Studies and Controlled Experimental Fairness
- **Definition**: An ablation or controlled addition study holds every variable constant (dataset split seed, transforms, backbone layers, optimizer, learning rate schedule, epoch budget) and changes **only one component** (`BaselineCNN` vs. `AttentionCNN`). Without those controls, you cannot know whether an accuracy difference came from the attention module or from a different learning rate or data split.

## 22. Reproducibility and Corruption Sensitivity
- **Deterministic Seeding**: Fixing seeds across `random`, `numpy`, and `torch` ensures that two runs on the same machine execute the exact same data split and mini-batch ordering.
- **Corruption Sensitivity**: Evaluating trained models across increasing severities $\sigma$ of Gaussian noise and Gaussian blur reveals how rapidly accuracy degrades when high-frequency pixel noise is added versus when spatial edges are smoothed out.
