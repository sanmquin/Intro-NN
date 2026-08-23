# Graph Shortest Path Extraction Benchmarks

This directory contains research tutorials, procedural dataset generators, and Transformer architectures for extracting direct shortest paths from algorithmic execution traces (goal-terminated Random Walk traces).

---

## 1. Complex Random Walk Dataset Specification

The procedural datasets (`graphs/data/graph_rw_easy_dataset.pt` and `graphs/data/graph_rw_hard_dataset.pt`) are generated using goal-terminated **Random Walk** traversals across two distinct graph topologies.

### Traversal Parameters & Sequence Bounds
- **Input Traversal Trace ($T$)**: Goal-terminated 1D Random Walk trace containing stochastic forward exploration and backtrack steps.
  - **Sequence Length ($K$)**: $100 \le K \le 200$ (`MAX_SRC_LEN = 200`)
  - The destination node $g$ appears **exactly once** at the final position ($t_K = g$).
- **Target Shortest Path ($P^*$)**: Direct shortest path connecting start node $s$ to destination node $g$.
  - **Sequence Length ($M$)**: $20 \le M \le 50$ (`MAX_TGT_LEN = 51` including `STOP_TOKEN`)
- **Vocabulary & Token Identifiers**:
  - Node Identifier Vocabulary: Tokens `0` through `49` ($V = 50$ randomized node IDs per sample).
  - Special Control Tokens: `PAD_TOKEN = 50`, `STOP_TOKEN = 51` (`VOCAB_SIZE = 52`).

### Dataset Flavors
1. **Easy Flavor (`graph_rw_easy_dataset.pt`)**:
   - Sparse tree/path-like topologies with minimal cycle complexity and low vertex connectivity.
2. **Hard Flavor (`graph_rw_hard_dataset.pt`)**:
   - Dense 2D grid/lattice topologies with abundant local cycles and multi-path connectivity.

---

## 2. Mechanics of a Good Plan vs. a Bad Plan

Sequential autoregressive rollout ($M \in [20, 50]$) over complex 1D Random Walk traces ($K \in [100, 200]$) evaluates the model's spatial planning and trajectory consistency.

### Good Plan Mechanics
- **Cross-Attention Alignment**: The decoder attends to the correct contextual representations in the encoded Random Walk memory $H_{src}$, identifying true forward edge transitions.
- **Valid Path Connectivity**: Each predicted step $p_m$ forms a valid edge $(p_{m-1}, p_m) \in E_G$ on the graph, terminating strictly at goal $g$.
- **Adjacency Compression**: The model successfully filters out redundant loop traversals and return steps embedded in $T$.

### Bad Plan Mechanics & Compounding Errors
- **Early Prefix Errors**: In long target sequences ($M \in [20, 50]$), an incorrect token choice at early step $m$ introduces an off-path node into the causal decoder context.
- **Compounding Error Propagation**: Once an invalid or off-path node is generated, the causal decoder state shifts into out-of-distribution space, leading to premature termination or hallucinated path loops.
- **Rollout Error Scaling**:
  $$P(\text{Exact Match}) = \prod_{m=1}^M P(p_m^* \mid p_{<m}^*, T) \approx (1 - \epsilon)^M$$

---

## 3. Notebook Configuration & Training Controls

`graphs/1.step_by_step_graph_shortest_path_tutorial.ipynb` includes explicit configuration controls in Cell 1 & Cell 5:

```python
config = {
    "dataset_flavor": "easy",       # Options: "easy" or "hard"
    "dataset_prefix": "rw_easy",    # Dataset name prefix for checkpoint filenames
    "model_size": "d64_l2_h4",      # Identifier for network size and architecture
    "restart_training": False,     # Set to True to bypass saved checkpoints and start fresh
    "run_full_training": False,    # Set to True to skip 'epochs_to_train' limit and run full 'total_epochs'
    "resume_training": True,       # Resumes from latest checkpoint if restart_training is False
    "total_epochs": 10000,
    "save_every": 1000,
    "validate_every": 50,
    "epochs_to_train": 20,         # Interactive execution chunk size
    "learning_rate": 1e-3,
    "batch_size": 64
}
```

### Key Configuration Flags & Multi-Checkpoint Storage
- **`dataset_flavor` & `dataset_prefix`**:
  - Selects between `graph_rw_easy_dataset.pt` and `graph_rw_hard_dataset.pt`.
- **`model_size` & Checkpoint Prefixing**:
  - Model checkpoints are automatically prefixed with dataset and network size tags (e.g., `ar_graph_rw_easy_d64_l2_h4_latest.pt` and `ar_graph_rw_easy_d64_l2_h4_epoch_1000.pt`), allowing side-by-side storage of checkpoints across different datasets and model architectures.

---

## 4. Directory Structure & Files

- `0.graph_dataset_and_topology_analysis_tutorial.ipynb`: Dataset generation notebook and topological characterization.
- `0.one_shot_graph_shortest_path_tutorial.ipynb`: One-Shot Non-Autoregressive Transformer tutorial.
- `1.step_by_step_graph_shortest_path_tutorial.ipynb`: Step-by-Step Autoregressive Graph Shortest Path Transformer tutorial.
- `2.mechanistic_interpretability_and_causal_analysis_tutorial.ipynb`: Mechanistic interpretability and causal analysis tutorial.
- `generate_data_notebook.py`: Programmatic generator for Notebook 0.
- `generate_notebook.py`: Programmatic generator for One-Shot Notebook.
- `generate_ar_notebook.py`: Programmatic generator for Autoregressive Notebook.
- `generate_mechanistic_notebook.py`: Programmatic generator for Mechanistic Analysis Notebook 2.
- `data/graph_rw_easy_dataset.pt`: Pre-generated Easy Random Walk dataset payload.
- `data/graph_rw_hard_dataset.pt`: Pre-generated Hard Random Walk dataset payload.
- `checkpoints/`: Local directory for model checkpoints.
- `charts/`: Output visualization figures.
