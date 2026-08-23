# Graph Shortest Path Extraction Benchmarks

This directory contains research tutorials, procedural dataset generators, and Transformer architectures for extracting direct shortest paths from algorithmic execution traces (goal-terminated Depth-First Search traces).

---

## 1. Complex Dataset Specification

The procedural dataset (`graphs/data/graph_dfs_dataset.pt`) is designed to evaluate transformer reasoning over deep search trees and multi-branch exploration traces.

### Traversal Parameters & Sequence Bounds
- **Input Traversal Trace ($T$)**: Goal-terminated 1D Depth-First Search (DFS) trace containing forward exploration and return/backtracking steps.
  - **Sequence Length ($K$)**: $30 \le K \le 50$ (`MAX_SRC_LEN = 50`)
  - The destination node $g$ appears **exactly once** at the final position ($t_K = g$).
- **Target Shortest Path ($P^*$)**: Direct shortest path connecting start node $s$ to destination node $g$.
  - **Sequence Length ($M$)**: $10 \le M \le 20$ (`MAX_TGT_LEN = 21` including `STOP_TOKEN`)
- **Vocabulary & Token Identifiers**:
  - Node Identifier Vocabulary: Tokens `0` through `39` ($V = 40$ randomized node IDs per sample).
  - Special Control Tokens: `PAD_TOKEN = 40`, `STOP_TOKEN = 41` (`VOCAB_SIZE = 42`).

### Node Backtraces & Induced Regressions Metric
During DFS traversal, whenever $t_k = t_{k-2}$, the transition represents a return step from dead-end or sub-branch node $t_{k-1}$ back to parent node $t_k$. We track two key metrics:
1. **Total Backtrace Count**: Total return steps in the trace.
2. **Node-Level Induced Regressions ($B(v)$)**: How many times node $v$ induced a backtrack/regression during traversal:
   $$B(v) = \sum_{k=3}^K \mathbb{I}\big(t_k = t_{k-2} \text{ and } t_{k-1} = v\big)$$

---

## 2. Mechanics of a Good Plan vs. a Bad Plan

Sequential autoregressive rollout ($M \in [10, 20]$) over complex 1D traversal traces ($K \in [30, 50]$) evaluates the model's spatial planning and trajectory consistency.

### Good Plan Mechanics
- **Cross-Attention Alignment**: The decoder attends to the correct contextual representations in the encoded DFS memory $H_{src}$, identifying true forward edge transitions.
- **Valid Path Connectivity**: Each predicted step $p_m$ forms a valid edge $(p_{m-1}, p_m) \in E_G$ on the graph, terminating strictly at goal $g$.
- **Adjacency Compression**: The model successfully filters out return steps ($t_k = t_{k-2}$) and dead-end subtrees embedded in $T$.

### Bad Plan Mechanics & Compounding Errors
- **Early Prefix Errors**: In long target sequences ($M \in [10, 20]$), an incorrect token choice at early step $m$ introduces an off-path node into the causal decoder context.
- **Compounding Error Propagation**: Once an invalid or off-path node is generated, the causal decoder state shifts into out-of-distribution space. Subsequent predictions fail to align with graph adjacencies, leading to premature termination or hallucinated path loops.
- **Rollout Error Scaling**: Because sequence match requires $M$ consecutive correct decisions, exact path match probability scales exponentially:
  $$P(\text{Exact Match}) = \prod_{m=1}^M P(p_m^* \mid p_{<m}^*, T) \approx (1 - \epsilon)^M$$
  With $M \ge 10$, even low token error rates $\epsilon \approx 0.05$ result in non-trivial rollout failure rates ($1 - 0.95^{15} \approx 53.7\%$).

---

## 3. Notebook Configuration & Training Controls

`graphs/1.step_by_step_graph_shortest_path_tutorial.ipynb` includes explicit configuration controls in Cell 5:

```python
config = {
    "restart_training": False,   # Set to True to bypass saved checkpoints and start fresh from epoch 1
    "run_full_training": False,  # Set to True to skip 'epochs_to_train' limit and run full 'total_epochs'
    "resume_training": True,     # Resumes from latest checkpoint if restart_training is False
    "total_epochs": 10000,
    "save_every": 1000,
    "validate_every": 50,
    "epochs_to_train": 20,       # Interactive execution chunk size
    "learning_rate": 1e-3,
    "batch_size": 64
}
```

### Key Configuration Flags
- **`restart_training`**:
  - `True`: Ignores existing checkpoints in `checkpoints/` and initializes model weights fresh from epoch 1.
  - `False`: Automatically attempts to load `ar_graph_transformer_latest.pt`.
- **`run_full_training`**:
  - `True`: Trains continuously up to `total_epochs` (e.g., 10,000 epochs) without stopping at `epochs_to_train`.
  - `False`: Runs an interactive chunk of `epochs_to_train` (e.g., 20 epochs) for local verification.
- **Periodic Validation & Checkpointing**:
  - Validation runs **strictly every 50 epochs** (`validate_every = 50`).
  - Model checkpoints are serialized every 1,000 epochs to Google Drive (`/content/drive/MyDrive/graph_checkpoints`) with local fallback (`checkpoints/`).

---

## 4. Directory Structure & Files

- `0.graph_dataset_and_topology_analysis_tutorial.ipynb`: Dataset generation notebook and topological characterization.
- `0.one_shot_graph_shortest_path_tutorial.ipynb`: One-Shot Non-Autoregressive Transformer tutorial.
- `1.step_by_step_graph_shortest_path_tutorial.ipynb`: Step-by-Step Autoregressive Graph Shortest Path Transformer tutorial.
- `generate_data_notebook.py`: Programmatic generator for Notebook 0.
- `generate_notebook.py`: Programmatic generator for One-Shot Notebook.
- `generate_ar_notebook.py`: Programmatic generator for Autoregressive Notebook.
- `data/graph_dfs_dataset.pt`: Pre-generated dataset payload.
- `checkpoints/`: Local directory for model checkpoints.
- `charts/`: Output visualization figures.
