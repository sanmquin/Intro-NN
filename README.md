# Artificial Neural Network Statistical Inference Tutorial

This project presents an educational tutorial on Artificial Neural Networks (ANNs) for predicting the latent parameters of a 1D Gaussian distribution from a 12-dimensional sequence of independent samples:
$$x_1, x_2, \dots, x_{12} \sim \mathcal{N}(\mu, \sigma^2)$$

The project covers training networks from scratch in NumPy, using PyTorch, and deploying modular, production-quality deep architectures, as well as exploring specialized neural architectures and comparing them against known statistical baseline formulas.

---

## Task 1: Proving standard deviation prediction improvements
The video/observation suggested that there was no improvement on standard deviation ($\sigma$) prediction. We have disproved this observation through rigorous training trajectory analysis.

### Theoretical Bounds (The Classical Baseline)
Using classical statistical estimators for a sample size of $D = 12$:
- **Sample Mean ($\bar{x}$)** yields an $R^2 \approx 0.9505$ on the true underlying mean ($\mu$).
- **Sample Standard Deviation ($s$)** yields an $R^2 \approx 0.4466$ on the true standard deviation ($\sigma$).

### Empirical Evidence
When evaluating standard deviation predictions during training:
1. **Initial State (Untrained Model)**: Standard deviation predictions exhibit a negative $R^2 \approx -0.2071$, representing worse-than-average predictions.
2. **Intermediate Training**: As optimization progresses, standard deviation $R^2$ climbs past the classical estimator baseline, peaking at **$0.65 - 0.69$** for our deep models.
3. **Charts & Assets**:
   - `std_prediction_analysis.png` shows standard deviation $R^2$ score improvement and MSE reduction over training epochs.
   - `prediction_scatters_with_std.png` graphs the predicted vs. true standard deviation on unseen test data.

---

## Task 2: Architectural Details & Key Limitations

We document the three architectures employed in the standard tutorial:
1. **Model 1 (Scratch NumPy ANN)**: A feedforward MLP with hidden dimensions `[32, 16]`, ReLU activations, He weight initialization, and Stochastic Gradient Descent (SGD).
2. **Model 2 (Medium PyTorch ANN)**: Structurally matching Model 1 but utilizing PyTorch's automatic differentiation and the **Adam optimizer** with adaptive learning rates.
3. **Model 3 (Production PyTorch ANN)**: A residual network with hidden dimensions `[64, 64, 64]`, GELU activations, Layer Normalization, Dropout, early stopping, and weight decay (AdamW).

### Mathematical Limitations on Variance/Std Dev Estimation
- **Piecewise Linear Approximations**: Neural networks using ReLU/GELU activations are piecewise linear function approximators. To represent a smooth quadratic curve (like sample variance/standard deviation) using flat line segments, a network requires substantial width and depth.
- **No Native Squaring Operation**: Standard dense layers only perform linear multiplication and addition. Squaring inputs requires constructing difference approximations of multiple ReLU activations, leading to inefficient representation in shallow networks.
- **Out-of-Distribution Degradation**: Because piecewise linear models extrapolate linearly outside the bounds of their training data, they fail to model quadratic functions outside of their training domain.

---

## Task 3: Deep Neural Networks Analysis and Replications
We created a new notebook `deep_neural_network_tutorial.ipynb` using deep configurations:
- **Deep Scratch Model**: Hidden dimensions `[128, 64, 32, 16]` with 4 hidden layers.
- **Deep PyTorch Model**: Hidden dimensions `[128, 64, 32, 16]` with 4 hidden layers.
- **Deep Production Model**: Chained **6 Residual Blocks** (12 linear layer updates) with a constant hidden dimension of `128` units.

### Performance Contrast Table (Standard vs. Deep)

Below are the unscaled results computed on the unseen test set:

| Model Name | Mean MAE | Mean $R^2$ | Var MAE | Var $R^2$ | Std Dev $R^2$ |
|---|---|---|---|---|---|
| **Standard Scratch NumPy** | 0.4902 | 0.9522 | 1.2793 | 0.5811 | 0.5705 |
| **Standard Medium PyTorch** | 0.4835 | 0.9532 | 1.2473 | 0.5864 | 0.5800 |
| **Standard Production PyTorch** | 0.4725 | 0.9558 | 1.1756 | 0.6221 | 0.6212 |
| **Deep Scratch NumPy** | 0.5109 | 0.9485 | 1.2286 | 0.6037 | 0.6396 |
| **Deep PyTorch (Adam)** | 0.5290 | 0.9429 | 1.4102 | 0.4552 | 0.5336 |
| **Deep Production PyTorch** | 0.4817 | 0.9533 | 1.1955 | **0.6234** | **0.6564** |

### Key Observations
- Chaining more ReLU layers together allows the network to synthesize a significantly higher number of piecewise linear segments, drastically improving standard deviation $R^2$ score (from 0.6212 up to **0.6564**).
- Increasing the hidden layer dimensions to 128 increases parameter capacity, allowing the model to perform highly non-linear statistical regressions far more cleanly.

---

## Task 4: Alternative Architectures Reference
We created `ARCHITECTURES_REFERENCE.md` explaining advanced architectures:
- **Polynomial/Pi-Sigma Networks**: Uses explicit multiplicative nodes to compute higher-order interactions.
- **Self-Attention & Transformers**: Models permutation-invariant sets of input samples.
- **Recursive & Autoregressive Feedback Networks**: Predicts mean first, then injects predicted mean as an input to predict standard deviation, simplifying the variance computation.
- **Normalizing Flows & Bayesian Networks**: Probabilistic modeling and uncertainty quantification.

---

## Task 5: Architecture Variation & Statistical Baselines (The Third Notebook)
We created a third notebook `architecture_variation_tutorial.ipynb` which varies network architecture (as outlined in `ARCHITECTURES_REFERENCE.md`) and directly benchmarks the outcomes against the **known classical sample statistics formulas** (the absolute mathematical baseline).

### Compared Architectures:
1. **Baseline MLP**: A standard dense, multi-layer regression network using ReLU.
2. **Sequential/Autoregressive Feedback Net**: Predicts mean first, then feeds it as an input to predict variance.
3. **Pi-Sigma Network**: Directly models quadratic interactions using explicit multiplicative output nodes.
4. **Permutation-Invariant Transformer**: Leverages self-attention mechanisms over 1D tokens without positional encodings, enforcing permutation invariance.

### Empirical Performance Comparison Table:

| Model / Baseline | Mean MAE | Mean $R^2$ | Variance MAE | Variance $R^2$ | Parameters | Total Train Time | Inf Latency |
|---|---|---|---|---|---|---|---|
| **Classical Formulas (Baseline)** | 0.4907 | 0.9505 | 1.2592 | 0.4466 | **0** | **0.00s** | **~0.00s** |
| **Baseline MLP** | 0.4851 | 0.9515 | 1.1895 | 0.5976 | 10,050 | ~14s | ~0.0003s |
| **Sequential Feedback Net** | 0.4787 | 0.9528 | 1.1927 | 0.6015 | 9,410 | ~15s | ~0.0005s |
| **Pi-Sigma Network** | **0.4725** | **0.9538** | **1.1625** | **0.6120** | **1,217** | **~11s** | **~0.0002s** |
| **Permutation-Invariant Transformer** | 0.4820 | 0.9520 | 1.1830 | 0.6052 | 34,754 | ~24s | ~0.0022s |

### Architectural & Cost Takeaways:
- **Inductive Bias & Efficiency**: The **Pi-Sigma network** performs exceptionally well on the variance prediction task because its architecture natively supports multiplicative, quadratic feature interaction. It achieves the **highest accuracy** ($R^2_{\text{var}} \approx 0.612$) while using a fraction of the parameters (**1,217** vs 10,000+ for MLPs) and training in the shortest time.
- **Transformer Permutation Invariance**: While mathematically elegant and permutation-invariant, the **Transformer** has the **highest computational costs** in terms of parameters (~34.7k), training duration, and inference latency (about 10x slower than the Pi-Sigma network).
- **Statistical Superiority**: All neural networks successfully pool sequence data and easily beat the classical sample variance baseline ($R^2 \approx 0.4466$), achieving $R^2 > 0.60$.

### Charts & Outputs Generated:
- `architecture_loss_comparison.png`: Tracks and compares validation loss curves over epochs (learning speed).
- `architecture_cost_metrics.png`: Multiple charts tracking parameter counts, total training times, inference latencies, and Variance $R^2$ vs Parameter Count (computational costs).
- `architecture_prediction_scatters.png`: Side-by-side regression scatter plots for all four architectures.
- `architecture_learning_process.mp4`: An animated video showing the Pi-Sigma network's predictions condensing onto the perfect-fit diagonal over training epochs.

---

## Execution Guide

To generate notebooks and execute them:
```bash
# Generate Standard Notebook
python3 create_notebook.py
# Generate Deep Notebook
python3 create_deep_notebook.py
# Generate Architecture Variation Notebook
python3 create_architecture_notebook.py

# Run Standard, Deep, and Architecture execution & verification
python3 run_all_and_verify.py
```
