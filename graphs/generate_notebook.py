import os
import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Introduction
    title_md = """# 0. One-Shot Graph Shortest Path Extraction Transformer
## Structural Graph Representation Learning from Algorithmic DFS Traversal Traces

### Executive Summary & Educational Motivation
Extracting spatial and structural information from raw algorithmic execution traces is a foundational challenge in neural algorithmic reasoning. When an agent or algorithm explores an unknown graph using **Depth-First Search (DFS)**, the resulting temporal trace contains both forward exploratory steps and return/backtracking steps.

In this tutorial, we demonstrate that a **One-Shot (Non-Autoregressive) Transformer** can process a complete 1D DFS traversal trace as an implicit graph specification and extract the direct shortest path between the start node and the destination node in a single parallel forward pass $\\mathcal{O}(1)$.

---

### Mathematical Derivation & Problem Formulation

#### 1. Graph Traversal and DFS Trace Representation
Let $G = (V, E)$ be an undirected connected graph where $V$ is a set of $N$ nodes ($20 \\le N \\le 35$) labeled with randomized tokens from a fixed vocabulary $\\mathcal{V}_{nodes} = \\{0, 1, \\dots, 39\\}$.

A Depth-First Search (DFS) algorithm starting at root node $s \\in V$ explores the graph and **terminates immediately upon reaching destination node $g$**. The generated sequential traversal trace is:
$$T = [t_1, t_2, \\dots, t_K]$$
where $t_1 = s$, $t_K = g$, and $g$ appears **exactly once** at position $K$. Each adjacent transition $(t_k, t_{k+1})$ represents either a forward exploration step along an edge $e \\in E$ or a return step (backtracking to a parent node from a dead end or fully explored branch).

The input sequence length $K$ satisfies $30 \\le K \\le 50$.

#### 2. Shortest Path Target Operator
Given the start node $s = t_1$ and destination node $g = t_K$, the target sequence is the shortest path sequence $P^*$:
$$P^* = [p_1^*, p_2^*, \\dots, p_M^*, \\text{STOP}]$$
where $p_1^* = s$, $p_M^* = g$, and $M$ is the minimum path length ($10 \\le M \\le 20$).

#### 3. Parallel One-Shot Cross-Attention Mechanism
The **One-Shot Transformer** utilizes $M_{max}=21$ learned positional query embeddings $Q \\in \\mathbb{R}^{M_{max} \\times d}$ to query the encoded DFS trace representations $H_{src} \\in \\mathbb{R}^{K \\times d}$:

$$\\text{Attention}(Q, K, V) = \\text{Softmax}\\left( \\frac{Q W_Q (H_{src} W_K)^T}{\\sqrt{d_k}} \\right) (H_{src} W_V)$$

All output tokens $p_1, p_2, \\dots, p_{M_{max}}$ are projected and predicted simultaneously in parallel.
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    # Cell 1: Setup & Environment
    cell1_code = """# Cell 1: Environment Setup and Random Seed Initialization

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Ensure directories exist
if os.path.basename(os.getcwd()) == "graphs":
    os.makedirs("../charts", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    LOCAL_DATA_PATH = "data/graph_dfs_dataset.pt"
else:
    os.makedirs("charts", exist_ok=True)
    os.makedirs("graphs/charts", exist_ok=True)
    LOCAL_DATA_PATH = "graphs/data/graph_dfs_dataset.pt"

torch.set_num_threads(1)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Environment initialized successfully. Running on: {device}")
"""
    cells.append(nbf.v4.new_code_cell(cell1_code))

    # Cell 2: Import Dataset
    cell2_md = """### Dataset Loading & Preprocessing
We load the pre-generated dataset containing $30 \\le K \\le 50$ DFS traversal traces and $10 \\le M \\le 20$ target shortest paths.
"""
    cells.append(nbf.v4.new_markdown_cell(cell2_md))

    cell2_code = """# Cell 2: Load Pre-generated Complex Dataset

if not os.path.exists(LOCAL_DATA_PATH):
    raise FileNotFoundError(f"Dataset file not found at '{LOCAL_DATA_PATH}'. Please run Notebook 0 first.")

dataset_payload = torch.load(LOCAL_DATA_PATH, weights_only=False)
train_raw = dataset_payload['train']
val_raw = dataset_payload['val']
test_raw = dataset_payload['test']

VOCAB_SIZE = dataset_payload.get('vocab_size', 42)
PAD_TOKEN = dataset_payload.get('pad_token', 40)
STOP_TOKEN = dataset_payload.get('stop_token', 41)
MAX_SRC_LEN = dataset_payload.get('max_src_len', 50)
MAX_TGT_LEN = dataset_payload.get('max_tgt_len', 21)

print(f"Loaded dataset: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")
print(f"Params: VOCAB_SIZE={VOCAB_SIZE}, MAX_SRC_LEN={MAX_SRC_LEN}, MAX_TGT_LEN={MAX_TGT_LEN}")
"""
    cells.append(nbf.v4.new_code_cell(cell2_code))

    # Cell 3: PyTorch Dataset Class
    cell3_md = """### PyTorch Dataset & DataLoader
Pads input DFS sequences up to `MAX_SRC_LEN=50` and target shortest paths up to `MAX_TGT_LEN=21`.
"""
    cells.append(nbf.v4.new_markdown_cell(cell3_md))

    cell3_code = """# Cell 3: Dataset Class Definition

class GraphDFSDataset(Dataset):
    def __init__(self, raw_data, max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_TGT_LEN):
        self.samples = []
        for item in raw_data:
            trace, sp, G, mapping = item[0], item[1], item[2], item[3]
            backtracks = item[4] if len(item) > 4 else 0
            node_backtraces = item[5] if len(item) > 5 else {}

            # Pad SRC
            src = list(trace) + [PAD_TOKEN] * (max_src_len - len(trace))
            src_mask = [0 if t != PAD_TOKEN else 1 for t in src]

            # Pad TGT
            tgt = list(sp) + [STOP_TOKEN]
            tgt = tgt + [PAD_TOKEN] * (max_tgt_len - len(tgt))

            self.samples.append((
                torch.tensor(src[:max_src_len], dtype=torch.long),
                torch.tensor(src_mask[:max_src_len], dtype=torch.bool),
                torch.tensor(tgt[:max_tgt_len], dtype=torch.long),
                trace,
                sp,
                G,
                backtracks,
                node_backtraces
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def graph_oneshot_collate_fn(batch):
    src = torch.stack([item[0] for item in batch])
    src_mask = torch.stack([item[1] for item in batch])
    tgt = torch.stack([item[2] for item in batch])
    traces = [item[3] for item in batch]
    sps = [item[4] for item in batch]
    graphs = [item[5] for item in batch]
    backtracks = [item[6] for item in batch]
    node_backtraces = [item[7] for item in batch]
    return src, src_mask, tgt, traces, sps, graphs, backtracks, node_backtraces

train_dataset = GraphDFSDataset(train_raw)
val_dataset = GraphDFSDataset(val_raw)
test_dataset = GraphDFSDataset(test_raw)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=graph_oneshot_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=graph_oneshot_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=graph_oneshot_collate_fn)

print(f"DataLoaders prepared. Batch size: 64")
"""
    cells.append(nbf.v4.new_code_cell(cell3_code))

    # Cell 4: Model Definition
    cell4_md = """### One-Shot Graph Transformer Architecture
The `OneShotGraphTransformer` uses $M_{max}=21$ learnable target query embeddings to extract target shortest paths in parallel.
"""
    cells.append(nbf.v4.new_markdown_cell(cell4_md))

    cell4_code = """# Cell 4: Model Definition

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class OneShotGraphTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=3, max_tgt_len=MAX_TGT_LEN):
        super(OneShotGraphTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.max_tgt_len = max_tgt_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.query_embeddings = nn.Parameter(torch.randn(1, max_tgt_len, embed_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.layer_norm_out = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, src_key_padding_mask=None):
        batch_size = src.size(0)

        src_emb = self.pos_encoder(self.token_embedding(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        queries = self.query_embeddings.repeat(batch_size, 1, 1)
        attn_out, attn_weights = self.cross_attn(
            query=queries, key=memory, value=memory, key_padding_mask=src_key_padding_mask
        )
        x = self.layer_norm(queries + attn_out)
        ffn_out = self.ffn(x)
        x = self.layer_norm_out(x + ffn_out)

        logits = self.fc_out(x)
        return logits, attn_weights

model = OneShotGraphTransformer().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"OneShotGraphTransformer initialized. Total Trainable Parameters: {total_params:,}")
"""
    cells.append(nbf.v4.new_code_cell(cell4_code))

    # Cell 5: Evaluation Helper Functions
    cell5_md = """### Evaluation Helper Functions
Evaluates loss, per-token accuracy, exact match, and graph path connectivity validity.
"""
    cells.append(nbf.v4.new_markdown_cell(cell5_md))

    cell5_code = """# Cell 5: Evaluation Functions

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    exact_matches = 0
    total_sequences = 0
    valid_paths = 0

    with torch.no_grad():
        for src, src_mask, tgt, traces, sps, graphs, backtracks, node_backtraces in dataloader:
            src, src_mask, tgt = src.to(device), src_mask.to(device), tgt.to(device)
            logits, _ = model(src, src_key_padding_mask=src_mask)

            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            total_loss += loss.item() * src.size(0)

            preds = torch.argmax(logits, dim=-1)

            for i in range(src.size(0)):
                pred_seq = preds[i].cpu().tolist()
                tgt_seq = tgt[i].cpu().tolist()

                clean_tgt = []
                for t in tgt_seq:
                    if t in (STOP_TOKEN, PAD_TOKEN):
                        break
                    clean_tgt.append(t)

                clean_pred = []
                for p in pred_seq:
                    if p in (STOP_TOKEN, PAD_TOKEN):
                        break
                    clean_pred.append(p)

                for t_idx in range(len(clean_tgt)):
                    total_tokens += 1
                    if t_idx < len(clean_pred) and clean_pred[t_idx] == clean_tgt[t_idx]:
                        correct_tokens += 1

                total_sequences += 1
                if clean_pred == clean_tgt:
                    exact_matches += 1

                G_eval = graphs[i]
                if len(clean_pred) >= 2 and clean_pred[0] == clean_tgt[0] and clean_pred[-1] == clean_tgt[-1]:
                    is_valid = True
                    for k in range(len(clean_pred) - 1):
                        u, v = clean_pred[k], clean_pred[k+1]
                        if not G_eval.has_edge(u, v):
                            is_valid = False
                            break
                    if is_valid:
                        valid_paths += 1

    mean_loss = total_loss / len(dataloader.dataset)
    token_acc = (correct_tokens / max(1, total_tokens)) * 100.0
    exact_match_acc = (exact_matches / max(1, total_sequences)) * 100.0
    path_validity_acc = (valid_paths / max(1, total_sequences)) * 100.0

    return mean_loss, token_acc, exact_match_acc, path_validity_acc

print("Evaluation functions loaded.")
"""
    cells.append(nbf.v4.new_code_cell(cell5_code))

    # Cell 6: Training Loop
    cell6_md = """### Interactive Model Training Loop
We train `OneShotGraphTransformer` for 20 epochs on the complex dataset.
"""
    cells.append(nbf.v4.new_markdown_cell(cell6_md))

    cell6_code = """# Cell 6: Model Training

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

num_epochs = 20
history = {
    'train_loss': [],
    'val_loss': [],
    'val_token_acc': [],
    'val_exact_match': [],
    'val_path_validity': []
}

print(f"Starting training for {num_epochs} epochs...")
start_train_time = time.time()

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0

    for src, src_mask, tgt, _, _, _, _, _ in train_loader:
        src, src_mask, tgt = src.to(device), src_mask.to(device), tgt.to(device)
        optimizer.zero_grad()

        logits, _ = model(src, src_key_padding_mask=src_mask)
        loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * src.size(0)

    scheduler.step()
    train_loss = running_loss / len(train_loader.dataset)

    val_loss, val_token_acc, val_exact_match, val_path_validity = evaluate(model, val_loader, device)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_token_acc'].append(val_token_acc)
    history['val_exact_match'].append(val_exact_match)
    history['val_path_validity'].append(val_path_validity)

    print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Val Token Acc: {val_token_acc:.2f}% | Val Exact Match: {val_exact_match:.2f}% | "
          f"Val Path Validity: {val_path_validity:.2f}%")

total_train_time = time.time() - start_train_time
print(f"\\nTraining complete in {total_train_time:.2f} seconds.")
"""
    cells.append(nbf.v4.new_code_cell(cell6_code))

    # Cell 7: Test Benchmark
    cell7_md = """### Test Dataset Benchmark Summary
Benchmark evaluation on unseen test traces ($30 \\le K \\le 50$, $10 \\le M \\le 20$).
"""
    cells.append(nbf.v4.new_markdown_cell(cell7_md))

    cell7_code = """# Cell 7: Test Benchmark Evaluation

test_loss, test_token_acc, test_exact_match, test_path_validity = evaluate(model, test_loader, device)

print("=" * 65)
print("             ONE-SHOT TRANSFORMER TEST BENCHMARK")
print("=" * 65)
print(f"{'Evaluation Metric':<35} | {'Model Score':<15}")
print("-" * 65)
print(f"{'Test Cross-Entropy Loss':<35} | {test_loss:<15.4f}")
print(f"{'Token Accuracy (%)':<35} | {test_token_acc:<15.2f}%")
print(f"{'Exact Path Match (%)':<35} | {test_exact_match:<15.2f}%")
print(f"{'Path Connectivity Validity (%)':<35} | {test_path_validity:<15.2f}%")
print("=" * 65)
"""
    cells.append(nbf.v4.new_code_cell(cell7_code))

    # Cell 8: Visualization
    cell8_md = """### Visualizations with Original Input Sequence Text & Chart
Generates analytical figures including original input trace sequences formatted in text and plotted on NetworkX graph layouts.
"""
    cells.append(nbf.v4.new_markdown_cell(cell8_md))

    cell8_code = """# Cell 8: Analytical Figures and Sample Visualizations

sns.set_theme(style="whitegrid", palette="mako")

# Chart 1: Training Trajectories
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cross-Entropy Loss', color=color, fontsize=12, fontweight='bold')
l1 = ax1.plot(range(1, num_epochs + 1), history['train_loss'], label='Train Loss', color='navy', linewidth=2, linestyle='--')
l2 = ax1.plot(range(1, num_epochs + 1), history['val_loss'], label='Val Loss', color='royalblue', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12, fontweight='bold')
l3 = ax2.plot(range(1, num_epochs + 1), history['val_token_acc'], label='Val Token Acc (%)', color='forestgreen', linewidth=2)
l4 = ax2.plot(range(1, num_epochs + 1), history['val_exact_match'], label='Val Exact Match (%)', color='darkgreen', linewidth=2, linestyle=':')
ax2.tick_params(axis='y', labelcolor=color)

lines = l1 + l2 + l3 + l4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9)

plt.title('One-Shot Graph Transformer: Training Trajectories', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig("../charts/graph_dfs_training_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig("charts/graph_dfs_training_curves.png", dpi=300, bbox_inches='tight')
else:
    plt.savefig("charts/graph_dfs_training_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig("graphs/charts/graph_dfs_training_curves.png", dpi=300, bbox_inches='tight')
plt.show()

# Chart 2: Sample Graph Shortest Path Extraction
sample_src, sample_mask, sample_tgt, sample_trace, sample_sp, G_sample, backtracks_sample, node_backtraces_sample = test_dataset[0]

model.eval()
with torch.no_grad():
    sample_src_b = sample_src.unsqueeze(0).to(device)
    sample_mask_b = sample_mask.unsqueeze(0).to(device)
    pred_logits, _ = model(sample_src_b, src_key_padding_mask=sample_mask_b)
    pred_tokens = torch.argmax(pred_logits[0], dim=-1).cpu().tolist()
    clean_pred = []
    for p in pred_tokens:
        if p in (STOP_TOKEN, PAD_TOKEN):
            break
        clean_pred.append(p)

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G_sample, seed=42)

nx.draw_networkx_nodes(G_sample, pos, node_color='lightgray', node_size=550)
nx.draw_networkx_edges(G_sample, pos, edge_color='silver', width=1.5)

sp_edges = [(sample_sp[i], sample_sp[i+1]) for i in range(len(sample_sp)-1)]
nx.draw_networkx_edges(G_sample, pos, edgelist=sp_edges, edge_color='#2b5c8f', width=3.5, label='True Shortest Path')

nx.draw_networkx_nodes(G_sample, pos, nodelist=[sample_sp[0]], node_color='limegreen', node_size=750, label='Start Node')
nx.draw_networkx_nodes(G_sample, pos, nodelist=[sample_sp[-1]], node_color='crimson', node_size=750, label='Goal Node')

labels = {node: str(node) for node in G_sample.nodes()}
nx.draw_networkx_labels(G_sample, pos, labels=labels, font_size=9, font_weight='bold')

# Text box with full original sequence
trace_str = f"Original Input DFS Trace (K={len(sample_trace)}):\\n" + ", ".join(map(str, sample_trace[:25])) + "\\n" + ", ".join(map(str, sample_trace[25:]))
sp_str = f"Target Shortest Path (M={len(sample_sp)}): {sample_sp}"
pred_str = f"Model Predicted Path: {clean_pred}"
backtrack_str = f"Total Backtracks: {backtracks_sample} | Node Regressions: {dict(list(node_backtraces_sample.items())[:5])}"

plt.gcf().text(0.12, 0.02, f"{trace_str}\\n{sp_str}\\n{pred_str}\\n{backtrack_str}",
               fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.title("One-Shot Shortest Path Extraction Layout", fontsize=13, fontweight='bold', pad=15)
plt.legend(scatterpoints=1, loc='upper left', frameon=True, facecolor='white')
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig("../charts/graph_dfs_sample_visualization.png", dpi=300, bbox_inches='tight')
    plt.savefig("charts/graph_dfs_sample_visualization.png", dpi=300, bbox_inches='tight')
else:
    plt.savefig("charts/graph_dfs_sample_visualization.png", dpi=300, bbox_inches='tight')
    plt.savefig("graphs/charts/graph_dfs_sample_visualization.png", dpi=300, bbox_inches='tight')
plt.show()

print("One-shot analytical figures successfully generated and saved.")
"""
    cells.append(nbf.v4.new_code_cell(cell8_code))

    # Cell 9: Summary
    cell9_md = """### Self-Reflection & Summary
1. **Hardened Traversal Parallel Extraction**:
   Evaluated One-Shot extraction on input sequence traces of length $30 \\le K \\le 50$ with shortest paths of length $10 \\le M \\le 20$.
2. **Backtrace Sensitivity**:
   Parallel query cross-attention routes over input DFS traces to extract spatial shortest paths amidst node backtraces and dead-ends.
"""
    cells.append(nbf.v4.new_markdown_cell(cell9_md))

    nb.cells = cells

    nb_path = "graphs/0.one_shot_graph_shortest_path_tutorial.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {nb_path}")

if __name__ == "__main__":
    build_notebook()
