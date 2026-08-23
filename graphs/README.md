# Graph Shortest Path Extraction Benchmarks

This directory contains research tutorials, procedural dataset generators, and Transformer architectures for extracting direct shortest paths from algorithmic execution traces (Depth-First Search and Random Walk traces).

---

## 1. Complex Dataset Specifications

### A. Depth-First Search (DFS) Dataset (`graphs/data/graph_dfs_dataset.pt`)
- **Input Traversal Trace ($T$)**: Goal-terminated 1D Depth-First Search trace ($30 \le K \le 50$, `MAX_SRC_LEN = 50`).
- **Target Shortest Path ($P^*$)**: Direct shortest path ($10 \le M \le 20$, `MAX_TGT_LEN = 21`).
- **Vocabulary**: 40 node tokens (`0`..`39`), `PAD_TOKEN = 40`, `STOP_TOKEN = 41` (`VOCAB_SIZE = 42`).
- **Splits**: Train (3,000), Validation (500), Test (500).

### B. Random Walk (RW) Datasets (`graphs/data/graph_rw_easy_dataset.pt` & `graph_rw_hard_dataset.pt`)
- **Input Traversal Trace ($T$)**: Goal-terminated 1D Random Walk trace ($100 \le K \le 200$, `MAX_SRC_LEN = 200`).
- **Target Shortest Path ($P^*$)**: Direct shortest path ($20 \le M \le 50$, `MAX_TGT_LEN = 51`).
- **Vocabulary**: 50 node tokens (`0`..`49`), `PAD_TOKEN = 50`, `STOP_TOKEN = 51` (`VOCAB_SIZE = 52`).
- **Flavors**:
  1. **Easy Flavor (`graph_rw_easy_dataset.pt`)**: Sparse, tree/path-like topologies with low vertex connectivity.
  2. **Hard Flavor (`graph_rw_hard_dataset.pt`)**: High connectivity 2D grid/lattice topologies with abundant local cycles.
- **Splits**: Train (3,000), Validation (500), Test (500) for both flavors.

---

## 2. Notebook Configuration & Multi-Checkpoint Controls

`graphs/1.step_by_step_graph_shortest_path_tutorial.ipynb` includes configuration controls in Cell 1 & Cell 5:

```python
config = {
    "dataset_flavor": "rw_easy",    # Options: "dfs", "rw_easy", or "rw_hard"
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

### Dataset Switching & Checkpoint Prefixing
- **`dataset_flavor`**: Selects between `"dfs"` (`graph_dfs_dataset.pt`), `"rw_easy"` (`graph_rw_easy_dataset.pt`), and `"rw_hard"` (`graph_rw_hard_dataset.pt`).
- **`dataset_prefix` & `model_size`**: Model checkpoints are dynamically named using prefixing tags:
  `ar_graph_{dataset_prefix}_{model_size}_latest.pt`
  `ar_graph_{dataset_prefix}_{model_size}_epoch_{epoch}.pt`
  This enables simultaneous storage and evaluation of checkpoints across different datasets and model network sizes.

---

## 3. Directory Structure & Files

- `0.graph_dataset_and_topology_analysis_tutorial.ipynb`: DFS dataset generation notebook and topological characterization.
- `0.random_walk_graph_dataset_tutorial.ipynb`: New Random Walk dataset generation notebook with easy/hard flavors and 3000/500/500 splits.
- `0.one_shot_graph_shortest_path_tutorial.ipynb`: One-Shot Non-Autoregressive Transformer tutorial.
- `1.step_by_step_graph_shortest_path_tutorial.ipynb`: Step-by-Step Autoregressive Graph Shortest Path Transformer tutorial supporting multi-dataset switching and prefix checkpoint storage.
- `2.mechanistic_interpretability_and_causal_analysis_tutorial.ipynb`: Mechanistic interpretability and causal analysis tutorial.
- `generate_data_notebook.py`: Programmatic generator for original DFS dataset notebook.
- `generate_rw_data_notebook.py`: Programmatic generator for new Random Walk dataset notebook.
- `generate_notebook.py`: Programmatic generator for One-Shot notebook.
- `generate_ar_notebook.py`: Programmatic generator for Autoregressive notebook.
- `generate_mechanistic_notebook.py`: Programmatic generator for Mechanistic Analysis notebook 2.
- `data/graph_dfs_dataset.pt`: DFS dataset payload (3000/500/500 splits).
- `data/graph_rw_easy_dataset.pt`: Easy Random Walk dataset payload (3000/500/500 splits).
- `data/graph_rw_hard_dataset.pt`: Hard Random Walk dataset payload (3000/500/500 splits).
- `checkpoints/`: Local directory for model checkpoints.
- `charts/`: Output visualization figures.
