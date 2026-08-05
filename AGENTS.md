# Repository Best Practices for Jupyter Notebooks

This document defines the strict quality, architectural, and educational standards for creating and maintaining Jupyter Notebooks within this repository. Notebooks in this codebase are designed to serve two simultaneous purposes:
1. **Clear Educational Tutorials**: Accessible to students and engineers who want to understand the core concepts from scratch.
2. **Rigorous Research Artifacts**: Containing full experimental details, mathematical derivations, reproducible results, and high-quality visualizations suitable for academic publication.

---

## 1. Core Principles

### 1.1 Complete Self-Containment (No Local Imports)
To ensure that notebooks can run seamlessly on external cloud platforms (such as Google Colab, Kaggle, or Paperspace) without requiring users to clone the entire repository or configure local paths:
* **All code must be self-contained within the notebook cells.**
* **Never import local helper functions, custom layers, dataset generators, or training loops from local Python files** (e.g., do not do `from labs.solver import ...`).
* Copy necessary model definitions, helper scripts, and statistical utilities directly into structural notebook cells, clearly documenting their purpose and mathematical form.

### 1.2 No Local Weight Loading
* Do not rely on locally saved state dicts (`.pt` or `.pth`) unless the notebook includes a complete fallback block to train the model from scratch if the file is not found.
* Whenever possible, define training loops with minimal epochs and sample datasets so that the model can be trained interactively in under 2 minutes during normal verification.

---

## 2. Notebook Structure and Flow

Every notebook must follow a sequential numeric prefix naming convention (e.g., `0.neural_network_tutorial.ipynb`, `1.transformer_tutorial.ipynb`) to fit into the overall educational curriculum, and must be organized with the following distinct sections:

1. **Title and Mathematical Derivations**:
   A clear title followed by a LaTeX-formatted mathematical formulation of the problem and the operators (e.g., SVD of attention, Markov chains, or partial visibility matrices).
2. **Environment & Dependency Installation**:
   A single cell at the top installing any required external packages (e.g., PyTorch, Matplotlib, Seaborn) using standard notebook escape commands.
3. **From-Scratch Component Implementation**:
   Implementation of the models, layers, datasets, and loss functions with comprehensive docstrings explaining the algorithmic steps.
4. **Interactive Training and Logging**:
   A transparent training loop that prints training metrics (loss, accuracy, learning rate) at regular, readable epochs.
5. **Rigorous Analysis and Plotting**:
   Detailed plotting cells that generate publication-quality figures, saving them in the `charts/` directory to keep the root directory clean.
6. **Self-Reflection & Summary**:
   A markdown cell summarizing the key takeaways, limitations, and direct links to the next notebook in the curriculum.

---

## 3. Visualization and Charting Standards

Visualizations are the core of our research. All plots must look clean and professional:
* **Style**: Use professional graphing themes (e.g., `sns.set_theme(style="whitegrid")` or custom stylesheet settings).
* **Labelling**: Every chart **must** have:
  - A descriptive title.
  - Clear axis labels with units (e.g., `Epochs`, `R² Score`, `Entropy (Nats)`).
  - Clear legends distinguishing baseline estimators from neural outputs.
* **Colors**: Avoid default high-contrast primary colors. Use coherent, modern color palettes (e.g., `viridis`, `mako`, `coolwarm`).
* **Storage**: Save all output figures programmatically into the `charts/` directory:
  ```python
  import os
  os.makedirs('charts', exist_ok=True)
  plt.savefig('charts/your_descriptive_filename.png', dpi=300, bbox_inches='tight')
  ```

---

## 4. Execution and Reproducibility

### 4.1 Seed Setting
Always set the random seeds for NumPy, PyTorch, and random utilities at the beginning of the notebook to guarantee that results are identical across execution environments:
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 4.2 Populated Execution Logs
**Never submit a blank or unexecuted notebook.**
Before saving and committing a notebook, execute it end-to-end to ensure all outputs, loss curves, and evaluation tables are visible. You can automate this execution from the command line:
```bash
jupyter nbconvert --to notebook --execute --inplace your_notebook.ipynb
```
This guarantees that any reader can instantly read the results, training history, and charts without having to run the code themselves.
