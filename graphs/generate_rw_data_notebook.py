import os
import nbformat as nbf

def build_rw_data_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Introduction
    title_md = """# 0. Stochastic Random Walk Graph Traversal Dataset Generation and Topology Analysis
## Empirical Characterization of Random Walk Traces and Underlying Graph Topologies

### Executive Summary & Educational Motivation
In neural algorithmic reasoning, extracting structured path information from non-deterministic or noisy traversal traces represents a challenging benchmark. Unlike deterministic search algorithms like Depth-First Search (DFS) which follow systematic stack-based recursion, a **Random Walk** agent explores an unknown graph by stochastically selecting adjacent edges at each step.

This notebook constructs a standardized procedural dataset of goal-terminated Random Walk traversal traces ($50 \\le K \\le 100$) and provides a thorough empirical analysis of the underlying graph topologies and target shortest paths ($15 \\le M \\le 25$).

---

### Mathematical Problem Formulation

#### 1. Graph Generation and Adjacency Structure
Let $G = (V, E)$ be an undirected, unweighted connected graph with $N$ nodes ($18 \\le N \\le 25$), where node identifiers are randomly permuted from a core vocabulary $\\mathcal{V}_{nodes} = \\{0, 1, \\dots, 24\\}$.

#### 2. Goal-Terminated Random Walk Traversal Trace
A Random Walk agent starting at root node $s \\in V$ traverses $G$ by transitioning to a uniformly sampled random neighbor $u_{k+1} \\sim \\text{Uniform}(\\text{Adj}(u_k))$ until discovering destination node $g \\in V$. The traversal **terminates immediately** upon reaching $g$, yielding the trace:
$$T = [t_1, t_2, \\dots, t_K]$$
where $t_1 = s$, $t_K = g$, $t_k \\neq g$ for all $1 \\le k < K$, and $50 \\le K \\le 100$.

#### 3. Core Vocabulary & Control Tokens
- **Core Node Vocabulary**: Tokens $\\{0, 1, \\dots, 24\\}$ (25 node IDs)
- **Special Tokens**: `PAD_TOKEN = 25`, `STOP_TOKEN = 26` (`VOCAB_SIZE = 27`)
- **Maximum Length Bounds**: `MAX_SRC_LEN = 100`, `MAX_TGT_LEN = 26`

#### 4. Shortest Path Target
The true shortest path $P^*$ between $s$ and $g$ in $G_T$ (the graph reconstructed from trace $T$) is represented as:
$$P^* = [p_1^*, p_2^*, \\dots, p_M^*]$$
where $p_1^* = s$, $p_M^* = g$, and $15 \\le M \\le 25$.
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
DATASET_PATH = os.path.join(DATA_DIR, "graph_rw_dataset.pt")
"""
    cells.append(nbf.v4.new_code_cell(cell1_code))

    # Cell 2: Procedural Dataset Generation Function
    cell2_md = """### Dataset Construction: Candidate-Filtered Goal-Terminated Random Walk Traces
To guarantee complex reasoning challenges with non-deterministic exploratory traces:
1. **Goal-Terminated Traversal**: Random Walk stops **immediately** upon discovering the goal node $g$, so $g$ appears **exactly once** at the final position (`trace[-1] == g`).
2. **Hardened Target Length Bounds**:
   - Random Walk Trace Length $K$: $50 \\le K \\le 100$ (`MAX_SRC_LEN = 100`)
   - Shortest Path Length $M$: $15 \\le M \\le 25$ (`MAX_TGT_LEN = 26`)
3. **Core Vocabulary**: 25 node IDs ($0 \\dots 24$), `PAD_TOKEN = 25`, `STOP_TOKEN = 26`, `VOCAB_SIZE = 27`.
"""
    cells.append(nbf.v4.new_markdown_cell(cell2_md))

    cell2_code = """# Cell 2: Procedural Candidate Sampling, Filtering, and Node Backtrace Metrics

VOCAB_SIZE = 27
PAD_TOKEN = 25
STOP_TOKEN = 26
MAX_SRC_LEN = 100
MAX_TGT_LEN = 26

def generate_single_rw_candidate(min_nodes=18, max_nodes=25, min_trace_len=50, max_trace_len=100, min_sp_len=15, max_sp_len=25):
    for attempt in range(500):
        sp_len = random.randint(min_sp_len, max_sp_len)
        G = nx.path_graph(sp_len)
        start = 0
        goal = sp_len - 1

        curr_nodes = sp_len
        max_n = random.randint(min_nodes, max_nodes)
        if curr_nodes < max_n:
            extra_n = max_n - curr_nodes
            for new_node in range(curr_nodes, curr_nodes + extra_n):
                parent = random.randint(0, new_node - 1)
                G.add_edge(parent, new_node)

        extra_edges = random.randint(1, 4)
        for _ in range(extra_edges):
            u, v = random.sample(list(G.nodes()), 2)
            if u != v:
                G.add_edge(u, v)

        curr = start
        trace = [start]
        max_steps = max_trace_len + 10
        steps = 0
        while curr != goal and steps < max_steps:
            neighbors = list(G.neighbors(curr))
            curr = random.choice(neighbors)
            trace.append(curr)
            steps += 1

        if curr != goal or trace.count(goal) != 1 or not (min_trace_len <= len(trace) <= max_trace_len):
            continue

        G_trace = nx.Graph()
        for i in range(len(trace) - 1):
            G_trace.add_edge(trace[i], trace[i+1])

        if not nx.has_path(G_trace, start, goal):
            continue

        sp = nx.shortest_path(G_trace, source=start, target=goal)

        if min_sp_len <= len(sp) <= max_sp_len:
            backtracks = 0
            node_backtraces = {node: 0 for node in G_trace.nodes()}
            for i in range(2, len(trace)):
                if trace[i] == trace[i-2]:
                    backtracks += 1
                    node_backtraces[trace[i-1]] += 1

            n = G_trace.number_of_nodes()
            vocab = list(range(25))
            perm = random.sample(vocab, n)
            mapping = {node: perm[i] for i, node in enumerate(G_trace.nodes())}

            perm_trace = [mapping[x] for x in trace]
            perm_sp = [mapping[x] for x in sp]
            G_perm = nx.relabel_nodes(G_trace, mapping)
            perm_node_backtraces = {mapping[k]: v for k, v in node_backtraces.items() if k in mapping}

            return perm_trace, perm_sp, G_perm, mapping, backtracks, perm_node_backtraces

    return None

def generate_filtered_dataset(target_samples=4000):
    dataset = []
    attempts = 0
    max_attempts = target_samples * 10
    while len(dataset) < target_samples and attempts < max_attempts:
        sample = generate_single_rw_candidate()
        attempts += 1
        if sample is not None:
            dataset.append(sample)
    return dataset

print("Generating 4,000 candidate-filtered Random Walk samples (src: 50-100, tgt: 15-25)...")
start_time = time.time()
raw_data = generate_filtered_dataset(4000)
print(f"Generated {len(raw_data)} samples in {time.time() - start_time:.2f} seconds.")

train_raw = raw_data[:3000]
val_raw = raw_data[3000:3500]
test_raw = raw_data[3500:4000]

print(f"Splits constructed: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")
"""
    cells.append(nbf.v4.new_code_cell(cell2_code))

    # Cell 3: Save Dataset to File
    cell3_md = """### Dataset Serialization
We serialize the raw processed dataset (including node traces, shortest path targets, NetworkX graph topologies, token mappings, and backtrace metrics) using PyTorch `torch.save()`.
"""
    cells.append(nbf.v4.new_markdown_cell(cell3_md))

    cell3_code = """# Cell 3: Save Processed Dataset to Data Directory

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
We compute key graph-theoretic and algorithmic trace properties across the dataset:
1. **Node & Edge Statistics**: Node counts, edge counts, and average degree.
2. **Sequence Bounds**: Random Walk trace lengths ($50 \\le K \\le 100$) and shortest path lengths ($15 \\le M \\le 25$).
3. **Path Compression Ratio**: $\\eta = \\text{Shortest Path Length } M / \\text{Random Walk Trace Length } K$.
"""
    cells.append(nbf.v4.new_markdown_cell(cell4_md))

    cell4_code = """# Cell 4: Compute Graph Topological Statistics

degrees = []
node_counts = []
edge_counts = []
trace_lengths = []
sp_lengths = []
compression_ratios = []
backtrack_counts = []

for trace, sp, G, mapping, backtracks, node_backtraces in raw_data:
    node_counts.append(G.number_of_nodes())
    edge_counts.append(G.number_of_edges())

    deg_list = [d for _, d in G.degree()]
    degrees.extend(deg_list)

    K = len(trace)
    M = len(sp)
    trace_lengths.append(K)
    sp_lengths.append(M)
    compression_ratios.append(M / K)
    backtrack_counts.append(backtracks)

print("=" * 70)
print("          RANDOM WALK GRAPH TRAVERSAL DATASET STATISTICS")
print("=" * 70)
print(f"{'Metric':<38} | {'Mean ± Std':<15} | {'[Min, Max]':<10}")
print("-" * 70)
print(f"{'Node Count (N)':<38} | {np.mean(node_counts):.2f} ± {np.std(node_counts):.2f}   | [{np.min(node_counts)}, {np.max(node_counts)}]")
print(f"{'Edge Count (|E|)':<38} | {np.mean(edge_counts):.2f} ± {np.std(edge_counts):.2f}   | [{np.min(edge_counts)}, {np.max(edge_counts)}]")
print(f"{'Average Node Degree (<k>)':<38} | {np.mean(degrees):.2f} ± {np.std(degrees):.2f}   | [{np.min(degrees)}, {np.max(degrees)}]")
print(f"{'Random Walk Trace Length (K) [50-100]':<38} | {np.mean(trace_lengths):.2f} ± {np.std(trace_lengths):.2f}   | [{np.min(trace_lengths)}, {np.max(trace_lengths)}]")
print(f"{'Shortest Path Length (M) [15-25]':<38} | {np.mean(sp_lengths):.2f} ± {np.std(sp_lengths):.2f}   | [{np.min(sp_lengths)}, {np.max(sp_lengths)}]")
print(f"{'Compression Ratio (M / K)':<38} | {np.mean(compression_ratios):.3f} ± {np.std(compression_ratios):.3f} | [{np.min(compression_ratios):.2f}, {np.max(compression_ratios):.2f}]")
print(f"{'Total Backtracks per Trace':<38} | {np.mean(backtrack_counts):.2f} ± {np.std(backtrack_counts):.2f}   | [{np.min(backtrack_counts)}, {np.max(backtrack_counts)}]")
print("=" * 70)
"""
    cells.append(nbf.v4.new_code_cell(cell4_code))

    # Cell 5: Publication Quality Figures
    cell5_md = """### Publication-Quality Analytical Visualizations
We generate and save comprehensive visualizations:
1. **Topology Distributions**: Histograms of node degrees, Random Walk trace lengths ($50 \\le K \\le 100$), and shortest path lengths ($15 \\le M \\le 25$) (`charts/graph_rw_topology_distributions.png`).
2. **Sample Graph Layouts**: NetworkX graph visualization overlaid with sample original input sequence text.
"""
    cells.append(nbf.v4.new_markdown_cell(cell5_md))

    cell5_code = """# Cell 5: Generate Publication-Quality Analytical Figures

sns.set_theme(style="whitegrid", palette="mako")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sns.histplot(degrees, discrete=True, kde=False, color='steelblue', ax=axes[0])
axes[0].set_title('Node Degree Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Node Degree (k)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)

sns.histplot(trace_lengths, discrete=True, kde=False, color='teal', ax=axes[1])
axes[1].set_title('Random Walk Trace Length (K) [50-100]', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Trace Length (Tokens)', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)

sns.histplot(sp_lengths, discrete=True, kde=False, color='darkblue', ax=axes[2])
axes[2].set_title('Shortest Path Length (M) [15-25]', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Path Length (Nodes)', fontsize=11)
axes[2].set_ylabel('Count', fontsize=11)

plt.tight_layout()
if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig("../charts/graph_rw_topology_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig("charts/graph_rw_topology_distributions.png", dpi=300, bbox_inches='tight')
else:
    plt.savefig("charts/graph_rw_topology_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig("graphs/charts/graph_rw_topology_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Sample Graph Layout
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for i in range(2):
    trace_sample, sp_sample, G_sample, mapping_sample, backtracks_sample, node_backtraces_sample = raw_data[i]
    pos = nx.spring_layout(G_sample, seed=42+i)

    nx.draw_networkx_nodes(G_sample, pos, ax=axes[i], node_color='lightgray', node_size=500)
    nx.draw_networkx_edges(G_sample, pos, ax=axes[i], edge_color='silver', width=1.5)

    sp_edges = [(sp_sample[k], sp_sample[k+1]) for k in range(len(sp_sample)-1)]
    nx.draw_networkx_edges(G_sample, pos, ax=axes[i], edgelist=sp_edges, edge_color='#2b5c8f', width=3.0)

    nx.draw_networkx_nodes(G_sample, pos, ax=axes[i], nodelist=[sp_sample[0]], node_color='limegreen', node_size=650)
    nx.draw_networkx_nodes(G_sample, pos, ax=axes[i], nodelist=[sp_sample[-1]], node_color='crimson', node_size=650)

    labels = {node: str(node) for node in G_sample.nodes()}
    nx.draw_networkx_labels(G_sample, pos, ax=axes[i], labels=labels, font_size=8, font_weight='bold')

    trace_text = f"Input Random Walk Trace (K={len(trace_sample)}):\\n" + ", ".join(map(str, trace_sample[:30])) + "...\\n" + ", ".join(map(str, trace_sample[-30:]))
    sp_text = f"Target Shortest Path (M={len(sp_sample)}): {sp_sample}"

    axes[i].text(0.02, 0.02, f"{trace_text}\\n{sp_text}",
                 transform=axes[i].transAxes, fontsize=8.5, verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    axes[i].set_title(f"Sample {i+1}: Random Walk K={len(trace_sample)} -> Shortest Path M={len(sp_sample)}", fontsize=11, fontweight='bold', pad=10)
    axes[i].axis('off')

plt.tight_layout()
if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig("../charts/graph_rw_sample_topologies.png", dpi=300, bbox_inches='tight')
    plt.savefig("charts/graph_rw_sample_topologies.png", dpi=300, bbox_inches='tight')
else:
    plt.savefig("charts/graph_rw_sample_topologies.png", dpi=300, bbox_inches='tight')
    plt.savefig("graphs/charts/graph_rw_sample_topologies.png", dpi=300, bbox_inches='tight')
plt.show()

print("Analytical figures successfully generated and saved.")
"""
    cells.append(nbf.v4.new_code_cell(cell5_code))

    # Cell 6: Reflection
    cell6_md = """### Self-Reflection & Summary
1. **Random Walk Benchmark Complexity**:
   Random walk traces present non-deterministic exploratory patterns, requiring models to ignore cyclic looping and extract the minimal shortest path.
2. **Parameters & Serialization**:
   Sequence bounds $50 \\le K \\le 100$ and $15 \\le M \\le 25$ with core vocabulary 25 ($V_{total} = 27$) are serialized to `graph_rw_dataset.pt`.
"""
    cells.append(nbf.v4.new_markdown_cell(cell6_md))

    nb.cells = cells

    nb_path = "graphs/0.random_walk_graph_dataset_tutorial.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {nb_path}")

if __name__ == "__main__":
    build_rw_data_notebook()
