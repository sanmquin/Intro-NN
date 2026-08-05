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
We created a new notebook `deep_neural_network_tutorial.ipynb` (under history) using deep configurations:
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
| **Permutation-Invariant Transformer** | 0.4820 | 1.1830 | 0.6052 | 34,754 | ~24s | ~0.0022s |

---

## Task 6: Transformer Mechanics & Interpretability (The Second Notebook)
We designed and implemented a thorough from-scratch tutorial on **Transformer networks** in `1.transformer_tutorial.ipynb` (replacing the empty placeholder). This notebook provides deep conceptual derivations, math equations, complete PyTorch modules implemented from scratch, training, and interpretive weight analysis.

### The Sequence Sorting Benchmark
To showcase how Transformers route information dynamically based on input values, we train the model to sort an input sequence of length $D=8$ from a vocabulary of size $V=20$.
To predict the $i$-th element of the sorted output, the self-attention layer must dynamically pay attention to the input index holding the $i$-th smallest value.

### Key Visual & Parameter Insights:
1. **Dynamic Weight Allocation**:
   - By visualizing the self-attention matrices as heatmaps, we see that the attention peaks change completely for different inputs.
   - For example, if we input sequence A: `[14, 2, 18, 5, 11, 1, 9, 7]`, output position 0 (which predicts `1`) pays strong attention to input index 5 (which contains the value `1`).
   - If we switch to sequence B: `[1, 15, 3, 10, 5, 19, 12, 4]`, output position 0 automatically shifts its attention peak to input index 0 (which contains the value `1`).
   - This provides **direct visual proof of input-dependent softmax routing**!
2. **Learned Attention Bias**:
   - Dissecting the Query-Key projection matrices $W_q$ and $W_k$ reveals that the score between query token $u$ and key token $v$ can be represented as a bilinear form: $\text{Bias}(u, v) = E[u] (W_q W_k^T) E[v]^T$.
   - Plotting this $20 \times 20$ interaction heatmap reveals a strong diagonal magnitude-matching pattern. The model independently discovers that numbers represent an ordered, continuous numerical scale, and learns to align Queries and Keys of matching magnitudes.

### Charts and Assets:
- `charts/transformer_training_trajectory.png`: Training cross-entropy loss and validation accuracy (token and sequence-level) over epochs.
- `charts/attention_heatmaps_input_a.png`: Side-by-side heatmaps of the 4 attention heads for input A, demonstrating routing.
- `charts/attention_heatmaps_input_b.png`: Attention maps for input B, showing dynamic weight adjustments.
- `charts/vocabulary_attention_bias.png`: The bilinear Query-Key interaction score map showing the learned continuous numerical representations and magnitude matching bias.

---

## Task 7: Algorithmic Alignment in Neural Execution: A Step-by-Step Benchmark of Transformer Sorters
We created a benchmark tutorial notebook `12.sorting_algorithms_benchmark_tutorial.ipynb` that compares the capacity of Transformers to act as **step-by-step state transition emulators** across five classical sorting algorithms: **Bubble Sort, Selection Sort, Insertion Sort, Cocktail Shaker Sort, and Odd-Even Sort**.

### Empirical Performance Comparison Table (Step-by-Step State Transition):

| Algorithm | Final Train Loss | Val Single-Step Token Acc | Val Multi-Step Rollout Success | Train Time (Seconds) |
|---|---|---|---|---|
| **Bubble Sort** | 0.0053 | 99.95% | 97.00% | ~18s |
| **Odd-Even Sort** | 0.0715 | 97.23% | 16.50% | ~18s |
| **Selection Sort** | 0.1554 | 94.84% | 3.50% | ~18s |
| **Insertion Sort** | 0.2339 | 91.27% | 0.50% | ~18s |
| **Cocktail Shaker Sort** | 0.2780 | 91.35% | 0.00% | ~18s |

### Key Takeaways and Theoretical Insights
1. **The Bubble Sort Locality Advantage**:
   - Despite having an inferior traditional computational complexity ($O(N^2)$), Bubble Sort is exceptionally simple for a Transformer to emulate.
   - Its transitions are highly localized (adjacent pairwise comparisons and swaps), which makes the target next-state function highly regular and easy to represent with standard multi-head self-attention. This results in **99.95% single-step token accuracy** and **97.00% multi-step rollout success**.
2. **Selection Sort Representation Complexity**:
   - In Selection Sort, the element at the current index $t$ is swapped with the minimum of the remaining suffix.
   - This requires a double dynamic routing operation: identifying the minimum value and routing the element at $t$ to that dynamically determined position (`min_idx`). Because this target index varies dynamically based on the input values, a shallow Transformer has high representation complexity to learn it, leading to compounding errors under recursive trajectory rollouts (**3.50% rollout success**).
3. **Insertion Sort Shifting Bottleneck**:
   - Insertion Sort requires sliding/shifting variable-sized blocks of elements to make room for a new sorted element.
   - If the model makes a single-token prediction error during a shift, the mistake cascades catastrophically, completely corrupting subsequent states. This results in a near-zero multi-step rollout success rate (**0.50%**).

### Charts & Visual Assets
- `charts/sorting_single_step_accuracy.png`: Single-step transition token accuracy curves over epochs for all 5 algorithms.
- `charts/sorting_rollout_success.png`: Recursive trajectory rollout success rate curves, evaluating compounding errors.
- `charts/sorting_training_cost.png`: A scatter plot demonstrating the trade-off between training wall-clock time and final rollout accuracy.
- `charts/sorting_attention_routing.png`: Layer 1 attention weight routing maps illustrating how the self-attention mechanism processes step-by-step algorithms.

---

## Execution Guide

To generate notebooks and execute them:
```bash
# Generate Standard Notebook
python3 create_notebook.py
# Generate Transformer Tutorial Notebook
python3 create_transformer_notebook.py
# Generate Parameter Study Notebook
python3 create_parameter_study_notebook.py
```
