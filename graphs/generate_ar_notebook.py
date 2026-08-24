import os
import nbformat as nbf

def build_ar_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Introduction
    title_md = """# 1. Step-by-Step Autoregressive Graph Shortest Path Transformer
## Sequential Causal Sequence-to-Sequence Modeling for Dense Algorithmic Traversal Traces

### Executive Summary & Educational Motivation
Extracting structural path information from complex, noisy algorithmic execution traces is a fundamental challenge in neural algorithmic reasoning. While **One-Shot (Non-Autoregressive)** models predict all path steps in parallel, **Step-by-Step Autoregressive** models generate the path token-by-token using causal self-attention and cross-attention over the encoded traversal trace.

In this tutorial, we implement an **Autoregressive Sequence-to-Sequence Graph Transformer** trained on hardened, candidate-filtered goal-terminated traversal traces ($30 \\le K \\le 50$, target $10 \\le M \\le 20$). The notebook supports dynamic dataset switching between **Dense Random Walk (`rw_dense`)**, **Random Walk (`rw`)**, and **Depth-First Search (`dfs`)** traversal traces, alongside multi-size network configurations (`small`, `base`, `large`) and dataset-prefixed checkpoint serialization.

In the **Dense Random Walk (`rw_dense`)** flavor:
- **4+ Minimum Node Connectivity**: Every node has a degree $k \\ge 4$ ($d_{\\text{avg}} \\ge 4.5$), presenting at least 4-way bifurcations at every step.
- **Loops & Cyclical Paths**: The underlying graph contains dense intersecting cycles, testing the model's ability to filter out non-optimal loops and extract direct shortest paths.

---

### Mathematical Problem Formulation

#### 1. Input Traversal Trace Encoding
Given an input traversal trace $T = [t_1, t_2, \\dots, t_K]$ ($30 \\le K \\le 50$) where $t_1 = s$ and $t_K = g$, the Transformer Encoder maps token embeddings into contextual representations:
$$H_{src} = \\text{Encoder}\\Big(E(T) + P(T)\\Big) \\in \\mathbb{R}^{K \\times d_{model}}$$

#### 2. Causal Autoregressive Decoding & Plan Mechanics
The target shortest path $P^* = [p_1^*, p_2^*, \\dots, p_M^*]$ ($10 \\le M \\le 20$) is predicted sequentially. At step $m$, given previous tokens $p_{<m}^* = [p_1^*, \\dots, p_{m-1}^*]$, the Decoder predicts:
$$P(p_m^* \\mid p_{<m}^*, T) = \\text{Softmax}\\Bigg(\\text{FC}\\bigg(\\text{Decoder}\\Big(E(p_{<m}^*) + P(p_{<m}^*), H_{src}, M_{causal}\\Big)\\bigg)\\Bigg)$$
where $M_{causal}$ is a causal triangular mask preventing lookahead to future target positions ($m' \\ge m$).

#### 3. Good Plan vs. Bad Plan Mechanics & Compounding Errors
In long-horizon sequential rollout ($M \\in [10, 20]$):
- **Good Plan**: Every predicted token $p_m$ is an adjacent valid node on $G$, maintaining valid path connectivity toward goal $g$.
- **Bad Plan & Regressions**: A single incorrect token $p_m$ shifts autoregressive context into out-of-distribution space, causing **compounding errors** where subsequent step predictions fail or hallucinate non-existent edges. The probability of sequence failure scales as $1 - (1 - \\epsilon)^M$.
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    # Cell 1: Environment Setup
    cell1_code = """# Cell 1: Environment Setup, Seeds, and Google Drive Configuration

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

# Ensure directories exist relative to CWD
if os.path.basename(os.getcwd()) == "graphs":
    os.makedirs("../charts", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    LOCAL_DATA_DIR = "data"
    LOCAL_CKPT_DIR = "checkpoints"
else:
    os.makedirs("charts", exist_ok=True)
    os.makedirs("graphs/charts", exist_ok=True)
    os.makedirs("graphs/checkpoints", exist_ok=True)
    LOCAL_DATA_DIR = "graphs/data"
    LOCAL_CKPT_DIR = "graphs/checkpoints"

torch.set_num_threads(1)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Google Drive Mount & Path Resolution
def setup_drive_paths():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        data_dir = "/content/drive/MyDrive/graph_data"
        ckpt_dir = "/content/drive/MyDrive/graph_checkpoints"
    except ImportError:
        data_dir = LOCAL_DATA_DIR
        ckpt_dir = LOCAL_CKPT_DIR

    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Data directory: {data_dir}")
    print(f"Checkpoints directory: {ckpt_dir}")
    return data_dir, ckpt_dir

DATA_DIR, CKPT_DIR = setup_drive_paths()
"""
    cells.append(nbf.v4.new_code_cell(cell1_code))

    # Cell 2: Dataset Loading & PyTorch Dataset Definition
    cell2_md = """### Dataset Loading & PyTorch Dataset Class
Loads the pre-generated dataset ($30 \\le K \\le 50$, $10 \\le M \\le 20$).
Supports switching dataset flavor (`rw_dense`, `rw`, or `dfs`) dynamically via `config["dataset_flavor"]`.
- `src`: Input traversal trace padded to `MAX_SRC_LEN=50` with `PAD_TOKEN=40`.
- `tgt`: Shortest path with `STOP_TOKEN=41` padded to `MAX_TGT_LEN=21` with `PAD_TOKEN=40`.
"""
    cells.append(nbf.v4.new_markdown_cell(cell2_md))

    cell2_code = """# Cell 2: Configurable Dataset Loading & PyTorch Dataset Definition

config = {
    "dataset_flavor": "rw_dense", # 'rw_dense' for Dense Random Walk, 'rw' for Random Walk, 'dfs' for Depth-First Search
    "model_size": "base",        # 'small', 'base', or 'large'
    "restart_training": False,   # Set to True to skip existing checkpoints and start fresh
    "run_full_training": False,  # Set to True to train for total_epochs ignoring epochs_to_train
    "resume_training": True,     # Resumes from latest checkpoint if restart_training is False
    "total_epochs": 10000,
    "save_every": 1000,
    "validate_every": 50,
    "epochs_to_train": 20,       # Interactive execution chunk size
    "learning_rate": 1e-3,
    "batch_size": 64
}

dataset_filename = f"graph_{config['dataset_flavor']}_dataset.pt"
DATASET_PATH = os.path.join(DATA_DIR, dataset_filename)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at '{DATASET_PATH}'. Please run dataset generator to create '{dataset_filename}'.")

print(f"Loading dataset payload from '{DATASET_PATH}' (Flavor: {config['dataset_flavor'].upper()})...")
dataset_payload = torch.load(DATASET_PATH, weights_only=False)
train_raw = dataset_payload['train']
val_raw = dataset_payload['val']
test_raw = dataset_payload['test']

VOCAB_SIZE = dataset_payload.get('vocab_size', 42)
PAD_TOKEN = dataset_payload.get('pad_token', 40)
STOP_TOKEN = dataset_payload.get('stop_token', 41)
MAX_SRC_LEN = dataset_payload.get('max_src_len', 50)
MAX_TGT_LEN = dataset_payload.get('max_tgt_len', 21)

class GraphARDataset(Dataset):
    def __init__(self, raw_data, max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_TGT_LEN):
        self.samples = []
        self.raw_data = raw_data
        for item in raw_data:
            trace, sp, G, mapping = item[0], item[1], item[2], item[3]
            backtracks = item[4] if len(item) > 4 else 0
            node_backtraces = item[5] if len(item) > 5 else {}

            src = list(trace) + [PAD_TOKEN] * (max_src_len - len(trace))
            src_mask = [False if t != PAD_TOKEN else True for t in src]

            tgt = list(sp) + [STOP_TOKEN]
            tgt = tgt + [PAD_TOKEN] * (max_tgt_len - len(tgt))
            tgt_mask = [False if t != PAD_TOKEN else True for t in tgt]

            self.samples.append((
                torch.tensor(src[:max_src_len], dtype=torch.long),
                torch.tensor(src_mask[:max_src_len], dtype=torch.bool),
                torch.tensor(tgt[:max_tgt_len], dtype=torch.long),
                torch.tensor(tgt_mask[:max_tgt_len], dtype=torch.bool),
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

def graph_ar_collate_fn(batch):
    src = torch.stack([item[0] for item in batch])
    src_mask = torch.stack([item[1] for item in batch])
    tgt = torch.stack([item[2] for item in batch])
    tgt_mask = torch.stack([item[3] for item in batch])
    traces = [item[4] for item in batch]
    sps = [item[5] for item in batch]
    graphs = [item[6] for item in batch]
    backtracks = [item[7] for item in batch]
    node_backtraces = [item[8] for item in batch]
    return src, src_mask, tgt, tgt_mask, traces, sps, graphs, backtracks, node_backtraces

train_dataset = GraphARDataset(train_raw)
val_dataset = GraphARDataset(val_raw)
test_dataset = GraphARDataset(test_raw)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=graph_ar_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=graph_ar_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=graph_ar_collate_fn)

print(f"Datasets loaded successfully ({config['dataset_flavor'].upper()}): Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
"""
    cells.append(nbf.v4.new_code_cell(cell2_code))

    # Cell 3: Model Architecture
    cell3_md = """### Step-by-Step Autoregressive Graph Transformer Architecture
The `AutoregressiveGraphTransformer` architecture supports dynamic network sizes (`small`, `base`, `large`):
1. **Positional Encoding Layer**: Sinusoidal positional embeddings.
2. **Encoder**: Multi-Head Self-Attention over padded input trace ($K \\le 50$).
3. **Causal Decoder**: Multi-Head Decoder with cross-attention and triangular causal mask.
4. **Output Head**: Linear projection to vocabulary logits $\\in \\mathbb{R}^{42}$.
"""
    cells.append(nbf.v4.new_markdown_cell(cell3_md))

    cell3_code = """# Cell 3: Model Architecture Definition & Network Sizing Configurations

MODEL_SIZES = {
    'small': {'embed_dim': 32, 'num_heads': 2, 'hidden_dim': 64, 'num_layers': 1},
    'base': {'embed_dim': 64, 'num_heads': 4, 'hidden_dim': 128, 'num_layers': 2},
    'large': {'embed_dim': 128, 'num_heads': 8, 'hidden_dim': 256, 'num_layers': 4}
}

model_size_params = MODEL_SIZES.get(config.get("model_size", "base"), MODEL_SIZES['base'])

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

class AutoregressiveGraphTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2):
        super(AutoregressiveGraphTransformer, self).__init__()
        self.embed_dim = embed_dim
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

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        src_emb = self.pos_encoder(self.token_embedding(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        tgt_emb = self.pos_encoder(self.token_embedding(tgt))
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        logits = self.fc_out(out)
        return logits, memory

    def solve_graph_autoregressive(self, src, src_key_padding_mask=None, max_tgt_len=MAX_TGT_LEN):
        self.eval()
        device = src.device
        batch_size = src.size(0)

        src_emb = self.pos_encoder(self.token_embedding(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # Start decoding with the start node (src[:, 0])
        curr_seqs = [[src[b, 0].item()] for b in range(batch_size)]
        finished = [False] * batch_size

        for step in range(max_tgt_len - 1):
            if all(finished):
                break

            curr_max_len = max(len(s) for s in curr_seqs)
            tgt_in = torch.full((batch_size, curr_max_len), PAD_TOKEN, dtype=torch.long, device=device)
            for b in range(batch_size):
                tgt_in[b, :len(curr_seqs[b])] = torch.tensor(curr_seqs[b], dtype=torch.long, device=device)

            tgt_mask = self.generate_square_subsequent_mask(curr_max_len, device)
            tgt_emb = self.pos_encoder(self.token_embedding(tgt_in))

            out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            logits = self.fc_out(out)

            for b in range(batch_size):
                if finished[b]:
                    continue
                last_idx = len(curr_seqs[b]) - 1
                next_tok = torch.argmax(logits[b, last_idx, :]).item()
                if next_tok in (STOP_TOKEN, PAD_TOKEN):
                    finished[b] = True
                else:
                    curr_seqs[b].append(next_tok)

        return curr_seqs

model = AutoregressiveGraphTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=model_size_params['embed_dim'],
    num_heads=model_size_params['num_heads'],
    hidden_dim=model_size_params['hidden_dim'],
    num_layers=model_size_params['num_layers']
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"AutoregressiveGraphTransformer ({config['dataset_flavor'].upper()}, Size: {config['model_size']}) initialized. Total Parameters: {total_params:,}")
"""
    cells.append(nbf.v4.new_code_cell(cell3_code))

    # Cell 4: Evaluation Functions
    cell4_md = """### Evaluation Helper Functions
Calculates:
1. Teacher-Forcing Cross-Entropy Loss & Token Accuracy.
2. Step-by-Step Rollout Exact Path Match (%).
3. Valid Graph Connectivity (%).
"""
    cells.append(nbf.v4.new_markdown_cell(cell4_md))

    cell4_code = """# Cell 4: Evaluation Helper Functions

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

def evaluate_model(model, dataloader, device, run_rollout=True):
    model.eval()
    total_loss = 0.0
    total_tf_tokens = 0
    correct_tf_tokens = 0

    exact_matches = 0
    total_sequences = 0
    valid_paths = 0

    with torch.no_grad():
        for src, src_mask, tgt, tgt_mask, traces, sps, graphs, backtracks, node_backtraces in dataloader:
            src, src_mask = src.to(device), src_mask.to(device)
            tgt, tgt_mask = tgt.to(device), tgt_mask.to(device)

            tgt_in = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            tgt_in_mask = tgt_mask[:, :-1]

            sz = tgt_in.size(1)
            causal_mask = model.generate_square_subsequent_mask(sz, device)

            logits, _ = model(
                src=src,
                tgt=tgt_in,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_in_mask,
                tgt_mask=causal_mask
            )

            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_label.reshape(-1))
            total_loss += loss.item() * src.size(0)

            preds_tf = torch.argmax(logits, dim=-1)
            valid_tokens = (tgt_label != PAD_TOKEN)
            correct_tf_tokens += ((preds_tf == tgt_label) & valid_tokens).sum().item()
            total_tf_tokens += valid_tokens.sum().item()

            if run_rollout:
                pred_seqs = model.solve_graph_autoregressive(src, src_key_padding_mask=src_mask)

                for i in range(src.size(0)):
                    total_sequences += 1
                    clean_pred = pred_seqs[i]
                    clean_tgt = list(sps[i])

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
    tf_acc = (correct_tf_tokens / max(1, total_tf_tokens)) * 100.0
    exact_match_acc = (exact_matches / max(1, total_sequences)) * 100.0 if run_rollout else 0.0
    path_validity_acc = (valid_paths / max(1, total_sequences)) * 100.0 if run_rollout else 0.0

    return mean_loss, tf_acc, exact_match_acc, path_validity_acc

print("Evaluation functions loaded.")
"""
    cells.append(nbf.v4.new_code_cell(cell4_code))

    # Cell 5: Training Loop with Dataset & Size Prefixed Checkpointing
    cell5_md = """### Configurable Training Loop with Dynamic Checkpoint Naming
Checkpoints are serialized using prefixes identifying dataset flavor and model size:
`ar_graph_transformer_{dataset_flavor}_{model_size}_latest.pt`
"""
    cells.append(nbf.v4.new_markdown_cell(cell5_md))

    cell5_code = """# Cell 5: Training Loop with Dynamic Dataset & Network Size Checkpoint Prefixes

ckpt_prefix = f"ar_graph_transformer_{config['dataset_flavor']}_{config['model_size']}"
latest_ckpt_path = os.path.join(CKPT_DIR, f"{ckpt_prefix}_latest.pt")

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

start_epoch = 1
history = {
    'train_loss': [],
    'val_epochs': [],
    'val_loss': [],
    'val_tf_acc': [],
    'val_exact_match': [],
    'val_path_validity': []
}

# Handle restart_training and resume_training configuration
if config.get("restart_training", False):
    print("Config 'restart_training' is True: Fresh initialization, skipping checkpoint loading.")
    start_epoch = 1
elif config.get("resume_training", True) and os.path.exists(latest_ckpt_path):
    try:
        print(f"Loading checkpoint from '{latest_ckpt_path}'...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"Resumed training from epoch {start_epoch}.")
    except Exception as e:
        print(f"Checkpoint incompatible ({e}). Starting fresh training from epoch 1...")
        start_epoch = 1
else:
    print(f"Starting training from scratch for configuration '{ckpt_prefix}'...")

# Determine end epoch based on run_full_training
if config.get("run_full_training", False):
    end_epoch = config["total_epochs"]
    print(f"Config 'run_full_training' is True: Running full training up to {end_epoch} epochs.")
else:
    end_epoch = min(start_epoch + config["epochs_to_train"] - 1, config["total_epochs"])
    print(f"Running epochs {start_epoch} to {end_epoch} (Config Total Target: {config['total_epochs']} epochs)...")

start_train_time = time.time()

for epoch in range(start_epoch, end_epoch + 1):
    model.train()
    running_loss = 0.0

    for src, src_mask, tgt, tgt_mask, _, _, _, _, _ in train_loader:
        src, src_mask = src.to(device), src_mask.to(device)
        tgt, tgt_mask = tgt.to(device), tgt_mask.to(device)

        tgt_in = tgt[:, :-1]
        tgt_label = tgt[:, 1:]
        tgt_in_mask = tgt_mask[:, :-1]

        sz = tgt_in.size(1)
        causal_mask = model.generate_square_subsequent_mask(sz, device)

        optimizer.zero_grad()
        logits, _ = model(
            src=src,
            tgt=tgt_in,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_in_mask,
            tgt_mask=causal_mask
        )

        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_label.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * src.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    history['train_loss'].append(train_loss)

    # Validation executed strictly every 50 epochs (or on final epoch of run)
    if epoch % config["validate_every"] == 0 or epoch == end_epoch:
        val_loss, val_tf_acc, val_exact_match, val_path_validity = evaluate_model(
            model, val_loader, device, run_rollout=True
        )
        history['val_epochs'].append(epoch)
        history['val_loss'].append(val_loss)
        history['val_tf_acc'].append(val_tf_acc)
        history['val_exact_match'].append(val_exact_match)
        history['val_path_validity'].append(val_path_validity)

        print(f"Epoch {epoch:04d}/{config['total_epochs']:04d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val TF Acc: {val_tf_acc:.2f}% | Rollout Exact Match: {val_exact_match:.2f}% | "
              f"Path Validity: {val_path_validity:.2f}%")
    else:
        if epoch % 5 == 0 or epoch == start_epoch:
            print(f"Epoch {epoch:04d}/{config['total_epochs']:04d} | Train Loss: {train_loss:.4f} | (Validation skipped for this epoch)")

    # Save checkpoint every 1,000 epochs (or on final epoch)
    if epoch % config["save_every"] == 0 or epoch == end_epoch:
        checkpoint_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'history': history
        }
        torch.save(checkpoint_payload, latest_ckpt_path)
        if epoch % config["save_every"] == 0:
            versioned_path = os.path.join(CKPT_DIR, f"{ckpt_prefix}_epoch_{epoch}.pt")
            torch.save(checkpoint_payload, versioned_path)
            print(f"Saved versioned checkpoint: '{versioned_path}'")

total_train_time = time.time() - start_train_time
print(f"\\nTraining chunk complete in {total_train_time:.2f} seconds.")
"""
    cells.append(nbf.v4.new_code_cell(cell5_code))

    # Cell 6: Test Benchmark
    cell6_md = """### Held-Out Test Set Evaluation
We evaluate the model on the unseen test dataset ($30 \\le K \\le 50$, $10 \\le M \\le 20$).
"""
    cells.append(nbf.v4.new_markdown_cell(cell6_md))

    cell6_code = """# Cell 6: Test Set Benchmark

test_loss, test_tf_acc, test_exact_match, test_path_validity = evaluate_model(
    model, test_loader, device, run_rollout=True
)

print("=" * 65)
print(f"  HELD-OUT TEST SET EVALUATION SUMMARY ({config['dataset_flavor'].upper()} - {config['model_size'].upper()})")
print("=" * 65)
print(f"{'Evaluation Metric':<35} | {'Model Score':<15}")
print("-" * 65)
print(f"{'Test Cross-Entropy Loss':<35} | {test_loss:<15.4f}")
print(f"{'Teacher-Forcing Token Acc (%)':<35} | {test_tf_acc:<15.2f}%")
print(f"{'Autoregressive Exact Match (%)':<35} | {test_exact_match:<15.2f}%")
print(f"{'Path Connectivity Validity (%)':<35} | {test_path_validity:<15.2f}%")
print("=" * 65)
"""
    cells.append(nbf.v4.new_code_cell(cell6_code))

    # Cell 7: Visualizations with Original Input Sequence Text & Chart
    cell7_md = """### Publication-Quality Visualizations & Sample Analysis
Generates:
1. **Training & Validation Loss Curves** (`charts/ar_graph_{dataset_flavor}_training_curves.png`).
2. **Sample Graph Shortest Path Rollout**: Includes the **original input sequence both in text format and in the visual chart layout**, alongside true vs. predicted path overlays and node backtrace counts (`charts/ar_graph_{dataset_flavor}_sample_visualization.png`).
"""
    cells.append(nbf.v4.new_markdown_cell(cell7_md))

    cell7_code = """# Cell 7: Generate Analytical Figures and Visual Layouts

sns.set_theme(style="whitegrid", palette="mako")

# Chart 1: Training Trajectories
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cross-Entropy Loss', color=color, fontsize=12, fontweight='bold')
epochs_range = range(1, len(history['train_loss']) + 1)
l1 = ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='navy', linewidth=2, linestyle='--')
if history['val_epochs']:
    l2 = ax1.plot(history['val_epochs'], history['val_loss'], label='Val Loss', color='royalblue', linewidth=2)
else:
    l2 = []
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12, fontweight='bold')
if history['val_epochs']:
    l3 = ax2.plot(history['val_epochs'], history['val_tf_acc'], label='Val TF Token Acc (%)', color='forestgreen', linewidth=2)
    l4 = ax2.plot(history['val_epochs'], history['val_exact_match'], label='Rollout Exact Match (%)', color='darkgreen', linewidth=2, linestyle=':')
    lines = l1 + l2 + l3 + l4
else:
    lines = l1
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9)

plt.title(f'Autoregressive Graph Transformer ({config["dataset_flavor"].upper()}): Training Trajectories', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
fig_filename_1 = f"ar_graph_{config['dataset_flavor']}_training_curves.png"
if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig(f"../charts/{fig_filename_1}", dpi=300, bbox_inches='tight')
    plt.savefig(f"charts/{fig_filename_1}", dpi=300, bbox_inches='tight')
else:
    plt.savefig(f"charts/{fig_filename_1}", dpi=300, bbox_inches='tight')
    plt.savefig(f"graphs/charts/{fig_filename_1}", dpi=300, bbox_inches='tight')
plt.show()

# Chart 2: Sample Graph Shortest Path Rollout with Original Sequence Text and Visual Layout
sample_src, sample_mask, sample_tgt, _, sample_trace, sample_sp, G_sample, backtracks_sample, node_backtraces_sample = test_dataset[0]

model.eval()
with torch.no_grad():
    sample_src_b = sample_src.unsqueeze(0).to(device)
    sample_mask_b = sample_mask.unsqueeze(0).to(device)
    pred_rollout = model.solve_graph_autoregressive(sample_src_b, src_key_padding_mask=sample_mask_b)[0]

plt.figure(figsize=(10, 7))
pos = nx.kamada_kawai_layout(G_sample)

# Draw base graph
nx.draw_networkx_nodes(G_sample, pos, node_color='lightgray', node_size=550)
nx.draw_networkx_edges(G_sample, pos, edge_color='silver', width=1.5, alpha=0.7)

# Highlight Shortest Path Edges
sp_edges = [(sample_sp[i], sample_sp[i+1]) for i in range(len(sample_sp)-1)]
nx.draw_networkx_edges(G_sample, pos, edgelist=sp_edges, edge_color='#2b5c8f', width=3.5, label='True Shortest Path')

# Highlight Start and Destination Nodes
nx.draw_networkx_nodes(G_sample, pos, nodelist=[sample_sp[0]], node_color='limegreen', node_size=750, label='Start Node')
nx.draw_networkx_nodes(G_sample, pos, nodelist=[sample_sp[-1]], node_color='crimson', node_size=750, label='Goal Node')

labels = {node: str(node) for node in G_sample.nodes()}
nx.draw_networkx_labels(G_sample, pos, labels=labels, font_size=9, font_weight='bold')

# Formatted original sequence text block
trace_str = f"Original Input {config['dataset_flavor'].upper()} Trace (K={len(sample_trace)}):\\n" + ", ".join(map(str, sample_trace[:25])) + "\\n" + ", ".join(map(str, sample_trace[25:]))
sp_str = f"Target Shortest Path (M={len(sample_sp)}): {sample_sp}"
pred_str = f"Autoregressive Predicted Path: {pred_rollout}"
deg_str = f"Nodes N={G_sample.number_of_nodes()} | Edges |E|={G_sample.number_of_edges()} | Min Deg={min(d for _, d in G_sample.degree())} | Avg Deg={sum(d for _, d in G_sample.degree())/G_sample.number_of_nodes():.2f}"

plt.gcf().text(0.12, 0.02, f"{trace_str}\\n{sp_str}\\n{pred_str}\\n{deg_str}",
               fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.title(f"Autoregressive Shortest Path Prediction Layout ({config['dataset_flavor'].upper()})", fontsize=13, fontweight='bold', pad=15)
plt.legend(scatterpoints=1, loc='upper left', frameon=True, facecolor='white')
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

fig_filename_2 = f"ar_graph_{config['dataset_flavor']}_sample_visualization.png"
if os.path.basename(os.getcwd()) == "graphs":
    plt.savefig(f"../charts/{fig_filename_2}", dpi=300, bbox_inches='tight')
    plt.savefig(f"charts/{fig_filename_2}", dpi=300, bbox_inches='tight')
else:
    plt.savefig(f"charts/{fig_filename_2}", dpi=300, bbox_inches='tight')
    plt.savefig(f"graphs/charts/{fig_filename_2}", dpi=300, bbox_inches='tight')
plt.show()

print("Publication-quality figures generated and saved.")
"""
    cells.append(nbf.v4.new_code_cell(cell7_code))

    # Cell 8: Plan Mechanics Analysis & Summary
    cell8_md = """### Self-Reflection & Mechanics of Good vs. Bad Plans

1. **Impact of Hardened Dense Trajectories ($30 \\le K \\le 50$, $10 \\le M \\le 20$, $d_{\\text{min}} \\ge 4$)**:
   In dense random walk traces, every state transition faces at least 4 candidate outgoing edges (4-way bifurcations) and abundant cyclical loops. The Transformer must learn to filter non-optimal loops and identify the true shortest path.
2. **Mechanics of a Good Plan vs. a Bad Plan**:
   - **Good Plan**: The autoregressive decoder successfully attends to valid cross-attention memory transitions, predicting step $p_m$ aligned with the graph adjacency matrix $A$.
   - **Bad Plan & Compounding Errors**: In long target sequences ($M \\in [10, 20]$), an early prediction error at step $m$ feeds an off-path token back into the causal decoder context. This causes compounding errors where the model loses spatial trajectory context and fails rollout exact match.
3. **Restart & Full Training Controls**:
   The notebook supports `"restart_training": True` to bypass saved checkpoints and `"run_full_training": True` to execute the full 10,000 epoch training schedule.
"""
    cells.append(nbf.v4.new_markdown_cell(cell8_md))

    nb.cells = cells

    nb_path = "graphs/1.step_by_step_graph_shortest_path_tutorial.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {nb_path}")

if __name__ == "__main__":
    build_ar_notebook()
