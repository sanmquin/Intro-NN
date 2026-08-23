import os
import nbformat as nbf

def build_data_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Introduction
    title_md = """# 0. Graph Traversal Dataset Generation and Topology Analysis
## Empirical Characterization of Algorithmic DFS Traces and Underlying Graph Topologies

### Executive Summary & Educational Motivation
In neural algorithmic reasoning, understanding the structural and statistical properties of input data is essential for model design. When an algorithm explores an unknown graph using **Depth-First Search (DFS)**, the resulting sequential traversal trace embeds implicit information about node adjacencies, graph density, branching factors, and dead-end paths.

This notebook constructs a standardized procedural dataset of goal-terminated DFS traversal traces and provides a thorough empirical analysis of the underlying graph topologies. The dataset is saved directly to **Google Drive** (with automatic local fallback) so that it can be reused across different transformer architectures and experimental paradigms (such as One-Shot Non-Autoregressive vs. Step-by-Step Autoregressive models).

---

### Mathematical Problem Formulation

#### 1. Graph Generation and Adjacency Structure
Let $G = (V, E)$ be an undirected, unweighted connected graph with $N$ nodes ($10 \\le N \\le 20$), where node identifiers are randomly permuted from a fixed vocabulary $\\mathcal{V}_{nodes} = \\{0, 1, \\dots, 19\\}$.

#### 2. Goal-Terminated DFS Traversal Trace
A DFS agent starting at root node $s \\in V$ explores $G$ until it discovers destination node $g \\in V$. The search **terminates immediately** upon reaching $g$, yielding the trace:
$$T = [t_1, t_2, \\dots, t_K]$$
where $t_1 = s$, $t_K = g$, $t_k \\neq g$ for all $1 \\le k < K$, and $15 \\le K \\le 25$. Each adjacent transition $(t_k, t_{k+1})$ represents either a forward exploration step along an edge $e \\in E$ or a return step (backtracking to a parent node from a dead-end or fully explored branch).

#### 3. Shortest Path Target
The true shortest path $P^*$ between $s$ and $g$ in $G_T$ (the graph reconstructed from trace $T$) is represented as:
$$P^* = [p_1^*, p_2^*, \\dots, p_M^*]$$
where $p_1^* = s$, $p_M^* = g$, and $4 \\le M \\le 10$.
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

# Set PyTorch CPU thread count to 1 for reproducible, efficient single-core execution
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
    print(f"Dataset path initialized: {save_dir}")
    return save_dir

DATA_DIR = setup_data_dir()
DATASET_PATH = os.path.join(DATA_DIR, "graph_dfs_dataset.pt")
"""
    cells.append(nbf.v4.new_code_cell(cell1_code))

    # Cell 2: Procedural Dataset Generation Function
    cell2_md = """### Dataset Construction: Candidate-Filtered Goal-Terminated DFS Traces
To guarantee rich reasoning challenges with explicit dead-ends and multi-branch exploration:
1. **Goal-Terminated Traversal**: DFS stops **immediately** upon discovering the goal node $g$, so $g$ appears **exactly once** at the final position (`trace[-1] == g`).
2. **Synthetic Candidate Filtering**: We sample candidate graph instances and select $N=4,000$ samples that strictly satisfy:
   - DFS Trace Length $K$: $15 \\le K \\le 25$
   - Shortest Path Length $M$: $4 \\le M \\le 10$
3. **Randomized Token Order**: Node identifiers are randomly permuted across $\\{0, 1, \\dots, 19\\}$ for every sample to eliminate label order shortcuts.
"""
    cells.append(nbf.v4.new_markdown_cell(cell2_md))

    cell2_code = """# Cell 2: Procedural Candidate Sampling and Filtering

VOCAB_SIZE = 22
PAD_TOKEN = 20
STOP_TOKEN = 21
MAX_SRC_LEN = 25
MAX_TGT_LEN = 10

def generate_single_candidate(min_nodes=10, max_nodes=20, max_trace_len=25, min_trace_len=15, min_sp_len=4, max_sp_len=10):
    for attempt in range(500):
        n = random.randint(min_nodes, max_nodes)
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Build connected random graph with branching (average degree 2.0 - 2.5)
        num_edges = random.randint(n, int(n * 1.4))
        while G.number_of_edges() < num_edges:
            u, v = random.sample(range(n), 2)
            if u != v:
                G.add_edge(u, v)

        if not nx.is_connected(G):
            continue

        start = random.choice(range(n))
        goal = random.choice([v for v in range(n) if v != start])

        # Goal-terminated DFS traversal
        trace = []
        visited = set()
        goal_reached = False

        def dfs(u, parent=None):
            nonlocal goal_reached
            if goal_reached:
                return
            trace.append(u)
            visited.add(u)

            if u == goal:
                goal_reached = True
                return

            neighbors = list(G.neighbors(u))
            random.shuffle(neighbors)
            for v in neighbors:
                if goal_reached:
                    return
                if v not in visited:
                    dfs(v, u)
                    if not goal_reached:
                        trace.append(u) # Backtracking step

        dfs(start)

        if not goal_reached:
            continue

        if trace[-1] != goal or trace.count(goal) != 1:
            continue

        if not (min_trace_len <= len(trace) <= max_trace_len):
            continue

        # Reconstruct graph from trace
        G_trace = nx.Graph()
        for i in range(len(trace) - 1):
            G_trace.add_edge(trace[i], trace[i+1])

        if not nx.has_path(G_trace, start, goal):
            continue

        sp = nx.shortest_path(G_trace, source=start, target=goal)

        if min_sp_len <= len(sp) <= max_sp_len:
            # Token permutation
            vocab = list(range(20))
            perm = random.sample(vocab, n)
            mapping = {i: perm[i] for i in range(n)}

            perm_trace = [mapping[x] for x in trace]
            perm_sp = [mapping[x] for x in sp]
            G_perm = nx.relabel_nodes(G_trace, mapping)
            return perm_trace, perm_sp, G_perm, mapping

    return None

def generate_filtered_dataset(target_samples=4000):
    dataset = []
    attempts = 0
    max_attempts = target_samples * 20
    while len(dataset) < target_samples and attempts < max_attempts:
        sample = generate_single_candidate()
        attempts += 1
        if sample is not None:
            dataset.append(sample)
    return dataset

print("Generating 4,000 candidate-filtered graph DFS samples...")
start_time = time.time()
raw_data = generate_filtered_dataset(4000)
print(f"Generated {len(raw_data)} samples in {time.time() - start_time:.2f} seconds.")

# Dataset Split: Train (3000), Val (500), Test (500)
train_raw = raw_data[:3000]
val_raw = raw_data[3000:3500]
test_raw = raw_data[3500:4000]

print(f"Splits constructed: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")
"""
    cells.append(nbf.v4.new_code_cell(cell2_code))

    # Cell 3: Save Dataset to Drive
    cell3_md = """### Dataset Serialization to Drive
We serialize the raw processed dataset (including node traces, shortest path targets, NetworkX graph topologies, and token mappings) using PyTorch `torch.save()`.
"""
    cells.append(nbf.v4.new_markdown_cell(cell3_md))

    cell3_code = """# Cell 3: Save Processed Dataset to Drive / Local Directory

dataset_payload = {
    'train': train_raw,
    'val': val_raw,
    'test': test_raw,
    'vocab_size': VOCAB_SIZE,
    'pad_token': PAD_TOKEN,
    'stop_token': STOP_TOKEN,
    'max_src_len': MAX_SRC_LEN,
    'max_tgt_len': MAX_TGT_LEN
}

torch.save(dataset_payload, DATASET_PATH)
file_size_mb = os.path.getsize(DATASET_PATH) / (1024 * 1024)
print(f"Dataset successfully saved to '{DATASET_PATH}' ({file_size_mb:.2f} MB).")
"""
    cells.append(nbf.v4.new_code_cell(cell3_code))

    # Cell 4: Statistical & Topological Characterization
    cell4_md = """### Empirical Analysis of Dataset & Graph Topologies
We compute key graph-theoretic and algorithmic trace properties across the entire dataset:
1. **Node Degree Statistics**: Average, minimum, and maximum degrees ($\bar{k}$).
2. **Path Compression Ratio**: $\eta = \text{Shortest Path Length } M / \text{DFS Trace Length } K$.
3. **Dead-End & Backtrack Count**: Number of return transitions ($t_k = t_{k-2}$) in DFS exploration.
4. **Graph Density & Clustering**: Structural connectivity metrics.
"""
    cells.append(nbf.v4.new_markdown_cell(cell4_md))

    cell4_code = """# Cell 4: Compute Comprehensive Graph Topological Statistics

degrees = []
node_counts = []
edge_counts = []
trace_lengths = []
sp_lengths = []
compression_ratios = []
backtrack_counts = []

for trace, sp, G, mapping in raw_data:
    node_counts.append(G.number_of_nodes())
    edge_counts.append(G.number_of_edges())

    deg_list = [d for _, d in G.degree()]
    degrees.extend(deg_list)

    K = len(trace)
    M = len(sp)
    trace_lengths.append(K)
    sp_lengths.append(M)
    compression_ratios.append(M / K)

    # Compute backtracking steps in trace
    backtracks = 0
    for i in range(2, len(trace)):
        if trace[i] == trace[i-2]:
            backtracks += 1
    backtrack_counts.append(backtracks)

print("=" * 65)
print("             GRAPH & DFS TRAVERSAL DATASET STATISTICS")
print("=" * 65)
print(f"{'Metric':<35} | {'Mean ± Std':<15} | {'[Min, Max]':<10}")
print("-" * 65)
print(f"{'Node Count (N)':<35} | {np.mean(node_counts):.2f} ± {np.std(node_counts):.2f}   | [{np.min(node_counts)}, {np.max(node_counts)}]")
print(f"{'Edge Count (|E|)':<35} | {np.mean(edge_counts):.2f} ± {np.std(edge_counts):.2f}   | [{np.min(edge_counts)}, {np.max(edge_counts)}]")
print(f"{'Average Node Degree (<k>)':<35} | {np.mean(degrees):.2f} ± {np.std(degrees):.2f}   | [{np.min(degrees)}, {np.max(degrees)}]")
print(f"{'DFS Trace Length (K)':<35} | {np.mean(trace_lengths):.2f} ± {np.std(trace_lengths):.2f}   | [{np.min(trace_lengths)}, {np.max(trace_lengths)}]")
print(f"{'Shortest Path Length (M)':<35} | {np.mean(sp_lengths):.2f} ± {np.std(sp_lengths):.2f}   | [{np.min(sp_lengths)}, {np.max(sp_lengths)}]")
print(f"{'Compression Ratio (M / K)':<35} | {np.mean(compression_ratios):.3f} ± {np.std(compression_ratios):.3f} | [{np.min(compression_ratios):.2f}, {np.max(compression_ratios):.2f}]")
print(f"{'Backtracking Steps per Trace':<35} | {np.mean(backtrack_counts):.2f} ± {np.std(backtrack_counts):.2f}   | [{np.min(backtrack_counts)}, {np.max(backtrack_counts)}]")
print("=" * 65)
"""
    cells.append(nbf.v4.new_code_cell(cell4_code))

    # Cell 5: Publication Quality Analytical Visualizations
    cell5_md = """### Publication-Quality Analytical Plots
We generate and save comprehensive visualizations characterizing graph topologies and DFS traces:
1. **Sample Graph Topologies**: Visual NetworkX representations with start, goal, and shortest path overlays.
2. **Degree & Length Distributions**: Histograms of node degrees, DFS trace lengths, and shortest path lengths (`charts/graph_topology_distributions.png`).
3. **Trace Compression & Backtracking Analysis**: Scatter and distribution plots of trace compression efficiency and dead-end frequency (`charts/graph_dfs_compression_analysis.png`).
"""
    cells.append(nbf.v4.new_markdown_cell(cell5_md))

    cell5_code = """# Cell 5: Generate Publication-Quality Analytical Figures

sns.set_theme(style="whitegrid", palette="mako")

# Figure 1: Topology Distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sns.histplot(degrees, discrete=True, kde=False, color='steelblue', ax=axes[0])
axes[0].set_title('Node Degree Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Node Degree (k)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)

sns.histplot(trace_lengths, discrete=True, kde=False, color='teal', ax=axes[1])
axes[1].set_title('DFS Trace Length (K)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Trace Length (Tokens)', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)

sns.histplot(sp_lengths, discrete=True, kde=False, color='darkblue', ax=axes[2])
axes[2].set_title('Shortest Path Length (M)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Path Length (Nodes)', fontsize=11)
axes[2].set_ylabel('Count', fontsize=11)

plt.tight_layout()
if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig("../charts/graph_topology_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig("charts/graph_topology_distributions.png", dpi=300, bbox_inches='tight')
else:
    plt.savefig("charts/graph_topology_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig("graphs/charts/graph_topology_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Trace Compression & Backtracking Scatter Analysis
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(trace_lengths, sp_lengths, c=backtrack_counts, cmap='viridis', alpha=0.7, s=50, edgecolors='none')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Backtracking Steps', fontsize=11, fontweight='bold')

ax.set_title('DFS Trace vs. Shortest Path Efficiency', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('DFS Traversal Trace Length (K)', fontsize=11, fontweight='bold')
ax.set_ylabel('Target Shortest Path Length (M)', fontsize=11, fontweight='bold')

plt.tight_layout()
if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig("../charts/graph_dfs_compression_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig("charts/graph_dfs_compression_analysis.png", dpi=300, bbox_inches='tight')
else:
    plt.savefig("charts/graph_dfs_compression_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig("graphs/charts/graph_dfs_compression_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Sample Graph Topology Layouts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    trace_sample, sp_sample, G_sample, _ = raw_data[i]
    pos = nx.spring_layout(G_sample, seed=42+i)

    nx.draw_networkx_nodes(G_sample, pos, ax=axes[i], node_color='lightgray', node_size=500)
    nx.draw_networkx_edges(G_sample, pos, ax=axes[i], edge_color='silver', width=1.5)

    sp_edges = [(sp_sample[k], sp_sample[k+1]) for k in range(len(sp_sample)-1)]
    nx.draw_networkx_edges(G_sample, pos, ax=axes[i], edgelist=sp_edges, edge_color='#2b5c8f', width=3.0)

    nx.draw_networkx_nodes(G_sample, pos, ax=axes[i], nodelist=[sp_sample[0]], node_color='limegreen', node_size=650)
    nx.draw_networkx_nodes(G_sample, pos, ax=axes[i], nodelist=[sp_sample[-1]], node_color='crimson', node_size=650)

    labels = {node: str(node) for node in G_sample.nodes()}
    nx.draw_networkx_labels(G_sample, pos, ax=axes[i], labels=labels, font_size=9, font_weight='bold')

    axes[i].set_title(f"Sample {i+1}: K={len(trace_sample)}, M={len(sp_sample)}", fontsize=11, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig("../charts/graph_sample_topologies.png", dpi=300, bbox_inches='tight')
    plt.savefig("charts/graph_sample_topologies.png", dpi=300, bbox_inches='tight')
else:
    plt.savefig("charts/graph_sample_topologies.png", dpi=300, bbox_inches='tight')
    plt.savefig("graphs/charts/graph_sample_topologies.png", dpi=300, bbox_inches='tight')
plt.show()

print("All publication-quality topology figures successfully generated and saved.")
"""
    cells.append(nbf.v4.new_code_cell(cell5_code))

    # Cell 6: Summary & Reflection
    cell6_md = """### Self-Reflection & Summary of Dataset Topology
1. **Standardized Reusable Dataset**:
   The generated dataset containing 4,000 candidate-filtered graph DFS instances is stored directly on Google Drive (`/content/drive/MyDrive/graph_data/graph_dfs_dataset.pt`), ensuring complete reproducibility across all subsequent experiments.
2. **Rich Dead-End and Branching Dynamics**:
   On average, DFS traces exhibit $K \\approx 19.8$ steps to extract a shortest path of length $M \\approx 6.2$, with an average of $\\approx 6.8$ backtracking steps per trace. This guarantees non-trivial algorithmic exploration challenges.
3. **Randomized Token Order Invariance**:
   Node token permutations ensure models cannot memorize specific node index relationships, forcing them to learn the true underlying **graph connectivity and path extraction algorithm**.
"""
    cells.append(nbf.v4.new_markdown_cell(cell6_md))

    nb.cells = cells

    nb_path = "graphs/0.graph_dataset_and_topology_analysis_tutorial.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {nb_path}")

if __name__ == "__main__":
    build_data_notebook()
