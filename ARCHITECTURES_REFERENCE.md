# Alternative Neural Network Architectures for Gaussian Parameter Inference

This reference document outlines distinct alternative deep learning architectures and methodologies designed to optimize parameter estimation tasks—specifically the non-linear estimation of sample variance ($\sigma^2$) and standard deviation ($\sigma$).

---

## 1. Polynomial and Quadratic Neural Networks

In standard feedforward networks, dense layers perform linear transformations ($Wx + b$) followed by piecewise linear activations like ReLU. To approximate a quadratic relationship like sample variance ($s^2 = \frac{1}{D-1}\sum (x_i - \bar{x})^2$), the network must synthesize quadratic curves using flat segments.

### Polynomial Networks / Pi-Sigma Networks
Polynomial neural networks introduce explicit multiplicative nodes. Instead of linear layers, they compute higher-order feature combinations:
- **Architecture**: A Pi-Sigma network uses a single hidden layer of linear summing units (Sigma) followed by product units (Pi) in the output layer.
- **Formulation**: The output is computed as:
  $$y = \prod_{k=1}^{K} (W_k^T X + b_k)$$
  For $K=2$, this directly models quadratic interactions ($x_i x_j$), allowing the network to compute variance in a single step with zero approximation error.
- **Benefits**: Directly captures the physics of quadratic relations, eliminating the need for deep piecewise approximations.

---

## 2. Transformer and Self-Attention Architectures

Rather than treating the input $X$ as a fixed 12D vector, we can treat it as a sequence of $D = 12$ distinct 1D tokens: $X = [x_1, x_2, \dots, x_{12}]$.

### Self-Attention Mechanism
- **Architecture**: Inputs are projected into Query ($Q$), Key ($K$), and Value ($V$) representations:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Variance Modeling via Attention**: The matrix product $Q K^T$ calculates pairwise multiplicative relationships between samples. When $Q$ and $K$ correspond to sample values, $Q K^T$ natively computes quadratic terms ($x_i x_j$), which are the raw building blocks of variance estimation.
- **Permutation Invariance**: Classical estimators of mean and variance are permutation-invariant. Transformers can achieve permutation invariance by omitting positional encodings. This forces the model to treat the inputs as a "bag of samples", matching the underlying statistical assumptions perfectly.

---

## 3. Recursive and Autoregressive Feedback Networks (Sequential Target Inference)

Standard networks perform joint regression, predicting both mean ($\hat{\mu}$) and variance ($\hat{\sigma}^2$) simultaneously. However, statistically, estimating variance is dependent on first knowing (or estimating) the mean:
$$s^2 = \frac{1}{D-1}\sum_{i=1}^{D} (x_i - \bar{x})^2$$

### Autoregressive Target Decoding (Recursive Networks with Target Feedback)
To model this dependency, we can use a **recursive feedback structure** where the network predicts targets sequentially. The output of the first stage is explicitly fed as an input to the second stage:

```
                          +-------------------+
Inputs (X) -------------> |  Mean Predictor   | ----> Predicted Mean (μ_pred)
   |                      +-------------------+            |
   |                                                       v
   +--------------------------------------------------> (Concat)
                                                           |
                                                           v
                                                  +-------------------+
                                                  | Variance Predictor| ----> Predicted Variance (σ²_pred)
                                                  +-------------------+
```

#### Detailed Computation Flow:
1. **Stage 1 (Mean Prediction)**:
   A dedicated network takes the sequence $X \in \mathbb{R}^{12}$ and predicts the mean:
   $$\hat{\mu} = f_{\text{mean}}(X; \theta_{\text{mean}})$$
2. **Recursive Input Prep**:
   The predicted mean $\hat{\mu}$ is concatenated with the raw input samples $X$ to form an augmented 13-dimensional representation:
   $$X_{\text{aug}} = [x_1, x_2, \dots, x_{12}, \hat{\mu}]^T \in \mathbb{R}^{13}$$
3. **Stage 2 (Variance/Std Dev Prediction)**:
   A second network (or recursive branch) takes $X_{\text{aug}}$ as input. This network can explicitly compute residuals $(x_i - \hat{\mu})$ because $\hat{\mu}$ is provided in the input vector:
   $$\hat{\sigma}^2 = f_{\text{var}}(X_{\text{aug}}; \theta_{\text{var}})$$
- **Why this works**: By providing the estimated mean directly to the variance predictor, the network does not have to implicitly compute and retain the mean inside its hidden layers. This isolates the non-linear squaring operation and reduces the optimization complexity of the variance network.

---

## 4. Normalizing Flows and Bayesian Neural Networks

Instead of performing point regression, these models treat parameter estimation as a probabilistic modeling task.

### Normalizing Flows
- **Mechanism**: Standard regression models predict the parameters $\mu$ and $\sigma^2$ directly using MSE loss. Normalizing flows instead model the entire probability distribution of the data:
  $$p_X(x) = p_Z(f(x)) \left| \det \frac{\partial f}{\partial x} \right|$$
  where $f(x)$ is a sequence of invertible bijector mappings.
- **Benefit**: Captures complex covariance structures and allows exact likelihood evaluation.

### Bayesian Neural Networks (BNNs)
- **Mechanism**: Parameters (weights and biases) are represented as distributions $p(W)$ rather than point values.
- **Benefit**: BNNs output both aleatoric uncertainty (inherent noise in the samples) and epistemic uncertainty (model uncertainty), preventing the network from making overconfident variance predictions in sparse data regions.
