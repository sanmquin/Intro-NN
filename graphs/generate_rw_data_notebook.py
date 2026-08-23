import os
import nbformat as nbf

def build_rw_data_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Introduction
    title_md = """# 0. Random Walk Graph Traversal Dataset Generation and Topology Analysis
## Empirical Characterization of Stochastic Random Walk Traversal Traces and Underlying Graph Topologies

### Executive Summary & Educational Motivation
In neural algorithmic reasoning, Depth-First Search (DFS) traces exhibit deterministic recursive backtracking. Replacing DFS with **Random Walk** traversals forces sequence models to navigate stochastic node transitions, requiring robust structural graph reasoning to extract direct shortest paths.

This notebook constructs standardized procedural datasets of goal-terminated **Random Walk** traversal traces across two distinct graph topological flavors:
1. **Easy Flavor (`graph_rw_easy_dataset.pt`)**: Sparse, tree/path-like topologies with low vertex connectivity and minimal cycle complexity.
2. **Hard Flavor (`graph_rw_hard_dataset.pt`)**: High connectivity topologies with abundant local cycles and multiple alternative routes.

Both dataset flavors are generated with exact standard splits: **3,000 training, 500 validation, and 500 test samples**.

---

### Mathematical Problem Formulation & Sequence Bounds

#### 1. Vocabulary and Token Boundaries
Node identifiers are randomly sampled and permuted from a core vocabulary of 50 tokens:
$$\\mathcal{V}_{nodes} = \\{0, 1, \\dots, 49\\}$$
Control tokens are designated as `PAD_TOKEN = 50` and `STOP_TOKEN = 51` (`VOCAB_SIZE = 52`).

#### 2. Sequence Bounds
- **Input Traversal Trace ($T$)**: Goal-terminated Random Walk trace sequence:
  $$100 \\le K \\le 200 \\quad (\\text{MAX\\_SRC\\_LEN} = 200)$$
- **Target Shortest Path ($P^*$)**: Minimum edge path connecting start node $s$ to goal $g$:
  $$20 \\le M \\le 50 \\quad (\\text{MAX\\_TGT\\_LEN} = 51 \\text{ including STOP token})$$

#### 3. Standard Splits
- **Train Set**: 3,000 samples
- **Validation Set**: 500 samples
- **Test Set**: 500 samples
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    # Cell 1: Environment Setup & Drive Mount
    cell1_code = """# Cell 1: Environment Setup, Seeds, and Google Drive Path Configuration

import os
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Ensure charts and data directories exist relative to repo root / working dir
if os.path.basename(os.getcwd()) == "graphs":
    os.makedirs("../charts", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    LOCAL_DATA_DIR = "data"
else:
    os.makedirs("charts", exist_ok=True)
    os.makedirs("graphs/charts", exist_ok=True)
    os.makedirs("graphs/data", exist_ok=True)
    LOCAL_DATA_DIR = "graphs/data"

# Set PyTorch CPU thread count to 1 for reproducible execution
torch.set_num_threads(1)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Google Drive Mount & Dataset Path Resolution
def setup_data_dir():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        save_dir = "/content/drive/MyDrive/graph_data"
    except ImportError:
        save_dir = LOCAL_DATA_DIR

    os.makedirs(save_dir, exist_ok=True)
    print(f"Dataset directory path initialized: {save_dir}")
    return save_dir

DATA_DIR = setup_data_dir()
DATASET_PATH_EASY = os.path.join(DATA_DIR, "graph_rw_easy_dataset.pt")
DATASET_PATH_HARD = os.path.join(DATA_DIR, "graph_rw_hard_dataset.pt")
"""
    cells.append(nbf.v4.new_code_cell(cell1_code))

    # Cell 2: Procedural Dataset Generation Function
    cell2_md = """### Dataset Construction: Candidate Sampling & Filtering for Easy vs Hard Random Walks
1. **Easy Graph Generator**: Sparse tree-like graph with 45-50 nodes and 1-3 extra edges.
2. **Hard Graph Generator**: Dense 2D grid/lattice with 50 nodes and local diagonal/cross edges creating multiple cycles.
3. **Random Walk Bounds**:
   - Random Walk Input Trace Length $K$: $100 \\le K \\le 200$
   - Target Shortest Path Length $M$: $20 \\le M \\le 50$
4. **Vocabulary Mapping**: Core vocabulary of 50 node tokens (`0` through `49`), permuted per graph instance.
"""
    cells.append(nbf.v4.new_markdown_cell(cell2_md))

    cell2_code = """# Cell 2: Procedural Candidate Sampling, Filtering, and Random Walk Generation

VOCAB_SIZE = 52
PAD_TOKEN = 50
STOP_TOKEN = 51
MAX_SRC_LEN = 200
MAX_TGT_LEN = 51

def generate_easy_graph(n=48):
    G = nx.path_graph(n)
    extra_edges = random.randint(1, 3)
    for _ in range(extra_edges):
        u, v = random.sample(range(n), 2)
        G.add_edge(u, v)
    return G

def generate_hard_graph(n=50):
    G = nx.grid_2d_graph(2, 25)
    G = nx.convert_node_labels_to_integers(G)
    for _ in range(random.randint(1, 3)):
        u = random.randint(0, 40)
        v = min(49, u + random.randint(2, 4))
        G.add_edge(u, v)
    return G

def generate_single_rw_candidate(flavor='easy', min_trace_len=100, max_trace_len=200, min_sp_len=20, max_sp_len=50):
    for attempt in range(20):
        if flavor == 'easy':
            n = random.randint(45, 50)
            G = generate_easy_graph(n)
            start = random.choice(range(10))
            goal = random.choice(range(35, n))
        else:
            n = 50
            G = generate_hard_graph(n)
            start = 0
            goal = 49

        if not nx.has_path(G, start, goal):
            continue

        sp = nx.shortest_path(G, source=start, target=goal)
        if not (min_sp_len <= len(sp) <= max_sp_len):
            continue

        dist_map = nx.single_source_shortest_path_length(G, goal)

        trace = [start]
        curr = start
        while curr != goal and len(trace) < max_trace_len + 50:
            neighbors = list(G.neighbors(curr))
            rem_dist = dist_map[curr]
            budget = 145 - len(trace)

            if len(trace) < min_trace_len - 10:
                non_goal_neighbors = [u for u in neighbors if u != goal]
                next_step = random.choice(non_goal_neighbors if non_goal_neighbors else neighbors)
            elif budget > rem_dist * 2:
                next_step = random.choice(neighbors)
            else:
                if random.random() < 0.70:
                    next_step = min(neighbors, key=lambda u: dist_map[u])
                else:
                    next_step = random.choice(neighbors)

            curr = next_step
            trace.append(curr)

        if curr != goal or not (min_trace_len <= len(trace) <= max_trace_len):
            continue

        # Backtrace count metric calculation (t_k == t_{k-2})
        backtracks = 0
        node_backtraces = {node: 0 for node in G.nodes()}
        for i in range(2, len(trace)):
            if trace[i] == trace[i-2]:
                backtracks += 1
                node_backtraces[trace[i-1]] += 1

        # Token permutation over vocabulary of 50 node IDs
        vocab = list(range(50))
        perm = random.sample(vocab, n)
        mapping = {i: perm[i] for i in range(n)}

        perm_trace = [mapping[x] for x in trace]
        perm_sp = [mapping[x] for x in sp]
        G_perm = nx.relabel_nodes(G, mapping)
        perm_node_backtraces = {mapping[k]: v for k, v in node_backtraces.items() if k in mapping}

        return perm_trace, perm_sp, G_perm, mapping, backtracks, perm_node_backtraces

    return None

def generate_rw_dataset(flavor='easy', target_samples=4000):
    dataset = []
    attempts = 0
    max_attempts = target_samples * 20
    while len(dataset) < target_samples and attempts < max_attempts:
        sample = generate_single_rw_candidate(flavor=flavor)
        attempts += 1
        if sample is not None:
            dataset.append(sample)
    return dataset

print("Generating Easy Random Walk dataset (4000 samples)...")
t0 = time.time()
raw_easy = generate_rw_dataset('easy', 4000)
print(f"Generated {len(raw_easy)} Easy samples in {time.time() - t0:.2f}s.")

print("Generating Hard Random Walk dataset (4000 samples)...")
t0 = time.time()
raw_hard = generate_rw_dataset('hard', 4000)
print(f"Generated {len(raw_hard)} Hard samples in {time.time() - t0:.2f}s.")
"""
    cells.append(nbf.v4.new_code_cell(cell2_code))

    # Cell 3: Save Datasets
    cell3_md = """### Dataset Serialization to File with Standard 3000/500/500 Splits
Both dataset payloads are structured into `train` (3,000), `val` (500), and `test` (500) splits.
"""
    cells.append(nbf.v4.new_markdown_cell(cell3_md))

    cell3_code = """# Cell 3: Save Processed Datasets with Standard Splits (3000 Train, 500 Val, 500 Test)

def build_split_payload(raw_data):
    return {
        'train': raw_data[:3000],
        'val': raw_data[3000:3500],
        'test': raw_data[3500:4000],
        'vocab_size': VOCAB_SIZE,
        'pad_token': PAD_TOKEN,
        'stop_token': STOP_TOKEN,
        'max_src_len': MAX_SRC_LEN,
        'max_tgt_len': MAX_TGT_LEN
    }

easy_payload = build_split_payload(raw_easy)
hard_payload = build_split_payload(raw_hard)

torch.save(easy_payload, DATASET_PATH_EASY)
torch.save(hard_payload, DATASET_PATH_HARD)

size_easy = os.path.getsize(DATASET_PATH_EASY) / (1024 * 1024)
size_hard = os.path.getsize(DATASET_PATH_HARD) / (1024 * 1024)

print(f"Easy Random Walk Dataset saved to '{DATASET_PATH_EASY}' ({size_easy:.2f} MB). Splits: Train={len(easy_payload['train'])}, Val={len(easy_payload['val'])}, Test={len(easy_payload['test'])}")
print(f"Hard Random Walk Dataset saved to '{DATASET_PATH_HARD}' ({size_hard:.2f} MB). Splits: Train={len(hard_payload['train'])}, Val={len(hard_payload['val'])}, Test={len(hard_payload['test'])}")
"""
    cells.append(nbf.v4.new_code_cell(cell3_code))

    # Cell 4: Statistical Analysis
    cell4_md = """### Empirical Analysis of Random Walk Datasets
Computes sequence bounds and backtrace statistics across Easy and Hard dataset flavors.
"""
    cells.append(nbf.v4.new_markdown_cell(cell4_md))

    cell4_code = """# Cell 4: Compute Random Walk Topology & Sequence Statistics

def print_dataset_stats(raw_data, label):
    trace_lens = [len(s[0]) for s in raw_data]
    sp_lens = [len(s[1]) for s in raw_data]
    backtracks = [s[4] for s in raw_data]

    print("=" * 70)
    print(f"         {label.upper()} RANDOM WALK DATASET STATISTICS")
    print("=" * 70)
    print(f"{'Metric':<38} | {'Mean ± Std':<15} | {'[Min, Max]':<10}")
    print("-" * 70)
    print(f"{'Input Trace Length (K) [100-200]':<38} | {np.mean(trace_lens):.2f} ± {np.std(trace_lens):.2f}   | [{np.min(trace_lens)}, {np.max(trace_lens)}]")
    print(f"{'Shortest Path Length (M) [20-50]':<38} | {np.mean(sp_lens):.2f} ± {np.std(sp_lens):.2f}   | [{np.min(sp_lens)}, {np.max(sp_lens)}]")
    print(f"{'Total Backtracks per Trace':<38} | {np.mean(backtracks):.2f} ± {np.std(backtracks):.2f}   | [{np.min(backtracks)}, {np.max(backtracks)}]")
    print("=" * 70)

print_dataset_stats(raw_easy, "EASY")
print_dataset_stats(raw_hard, "HARD")
"""
    cells.append(nbf.v4.new_code_cell(cell4_code))

    # Cell 5: Visualizations
    cell5_md = """### Analytical Figures: Distributions
Visualizing input trace length and target shortest path length distributions.
"""
    cells.append(nbf.v4.new_markdown_cell(cell5_md))

    cell5_code = """# Cell 5: Publication-Quality Visualizations

sns.set_theme(style="whitegrid", palette="mako")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.histplot([len(s[0]) for s in raw_easy], discrete=True, color='teal', ax=axes[0], label='Easy')
sns.histplot([len(s[0]) for s in raw_hard], discrete=True, color='darkred', ax=axes[0], label='Hard')
axes[0].set_title('Random Walk Trace Length (K) [100-200]', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sequence Length (Tokens)')
axes[0].legend()

sns.histplot([len(s[1]) for s in raw_easy], discrete=True, color='teal', ax=axes[1], label='Easy')
sns.histplot([len(s[1]) for s in raw_hard], discrete=True, color='darkred', ax=axes[1], label='Hard')
axes[1].set_title('Target Shortest Path Length (M) [20-50]', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Path Length (Nodes)')
axes[1].legend()

plt.tight_layout()
if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig("../charts/graph_rw_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig("charts/graph_rw_distributions.png", dpi=300, bbox_inches='tight')
else:
    plt.savefig("charts/graph_rw_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig("graphs/charts/graph_rw_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

print("Random walk dataset analysis figures generated and saved.")
"""
    cells.append(nbf.v4.new_code_cell(cell5_code))

    # Cell 6: Summary
    cell6_md = """### Summary & Reflection
1. **Random Walk Traversals**: Stochastic goal-terminated random walk traces ($100 \\le K \\le 200$) with target shortest paths ($20 \\le M \\le 50$).
2. **Two Flavors**: Easy (tree/path-like sparse graphs) and Hard (dense grid with cycles).
3. **Core Vocabulary**: 50 node tokens + 2 control tokens (`VOCAB_SIZE = 52`).
4. **Standard Splits**: Exactly 3000 Train, 500 Val, and 500 Test samples.
"""
    cells.append(nbf.v4.new_markdown_cell(cell6_md))

    nb.cells = cells

    nb_path = "graphs/0.random_walk_graph_dataset_tutorial.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {nb_path}")

if __name__ == "__main__":
    build_rw_data_notebook()
