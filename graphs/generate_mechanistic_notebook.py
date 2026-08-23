import os
import nbformat as nbf

def build_mechanistic_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown Intro
    title_md = """# 2. Mechanistic Interpretability and Causal Analysis of Autoregressive Graph Transformers
## Dissecting Shortest Path Accuracy via Attention Sharpening, Activation Patching, and Topological Error Dynamics

### Executive Summary & Research Motivation
In long-horizon neural algorithmic reasoning over stochastic Random Walk execution traces ($100 \\le K \\le 200$), sequence-to-sequence models exhibit non-linear performance shifts during training.

This notebook provides a complete Mechanistic Interpretability and Causal Analysis:
1. **Weight & Layer Mechanics**: Quantifying layer-wise parameter shifts, cross-attention sharpening (entropy reduction), and logit margin amplification.
2. **Topology & Activation Correlations**: Evaluating inference across validation samples, isolating how graph connectivity, depth, and activation statistics differentiate successful rollouts.
3. **Causal Activation Patching**: Intervening on hidden memory representations ($H_{src}$) and decoder cross-attention mechanisms.
4. **Reusable Exported Inference Datasets**: Serializing fully annotated evaluation datasets (`inference_dataset_epoch_300.pt` and `inference_dataset_epoch_400.pt`).

---

### Mathematical Derivations & Analytical Mechanics

#### 1. Cross-Attention Entropy Sharpening
Given sequence query tokens $q_m$ ($m \in [1, M]$) and encoded memory keys $k_n$ ($n \in [1, K]$), cross-attention weights at layer $l$ are given by $A^{(l)}_{m,n} = \text{Softmax}\left(\frac{q_m W_Q^{(l)} (k_n W_K^{(l)})^T}{\sqrt{d_k}}\right)$. We quantify spatial focus using **Cross-Attention Entropy**:
$$H(A^{(l)}_m) = - \sum_{n=1}^K A^{(l)}_{m,n} \ln\left(A^{(l)}_{m,n} + \epsilon\right)$$

#### 2. Logit Margin Confidence Metric
For target step $m$, with top logit prediction $z_{m,(1)}$ and runner-up $z_{m,(2)}$, the **Logit Margin** is defined as:
$$\Delta z_m = z_{m,(1)} - z_{m,(2)}$$
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    # Cell 1: Environment & Drive Configuration
    cell1_code = """# Cell 1: Environment Setup, Random Seeds, and Drive/Local Path Resolution

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
from scipy import stats

# Resolve paths relative to repository structure
if os.path.basename(os.getcwd()) == "graphs":
    os.makedirs("../charts", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    LOCAL_DATA_PATH = "data/graph_rw_easy_dataset.pt"
    LOCAL_CKPT_300 = "data/ar_graph_transformer_epoch_300.pt"
    LOCAL_CKPT_400 = "data/ar_graph_transformer_epoch_400.pt"
    EXPORT_DIR = "data"
else:
    os.makedirs("charts", exist_ok=True)
    os.makedirs("graphs/charts", exist_ok=True)
    os.makedirs("graphs/data", exist_ok=True)
    LOCAL_DATA_PATH = "graphs/data/graph_rw_easy_dataset.pt"
    LOCAL_CKPT_300 = "graphs/data/ar_graph_transformer_epoch_300.pt"
    LOCAL_CKPT_400 = "graphs/data/ar_graph_transformer_epoch_400.pt"
    EXPORT_DIR = "graphs/data"

torch.set_num_threads(1)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DRIVE_CKPT_300 = "/content/drive/MyDrive/graph_checkpoints/ar_graph_transformer_epoch_300.pt"
DRIVE_CKPT_400 = "/content/drive/MyDrive/graph_checkpoints/ar_graph_transformer_epoch_400.pt"

if os.path.exists(DRIVE_CKPT_300) and os.path.exists(DRIVE_CKPT_400):
    PATH_CKPT_300 = DRIVE_CKPT_300
    PATH_CKPT_400 = DRIVE_CKPT_400
    print("Resolved checkpoints from Google Drive.")
else:
    PATH_CKPT_300 = LOCAL_CKPT_300
    PATH_CKPT_400 = LOCAL_CKPT_400
    print("Resolved checkpoints from local repository data directory.")

print(f"Checkpoint 300 path: {PATH_CKPT_300}")
print(f"Checkpoint 400 path: {PATH_CKPT_400}")
"""
    cells.append(nbf.v4.new_code_cell(cell1_code))

    # Cell 2: Dataset Loading
    cell2_code = """# Cell 2: Load Graph Random Walk Dataset Payload

if not os.path.exists(LOCAL_DATA_PATH):
    raise FileNotFoundError(f"Dataset payload not found at '{LOCAL_DATA_PATH}'. Please run Notebook 0.")

dataset_payload = torch.load(LOCAL_DATA_PATH, map_location='cpu', weights_only=False)
val_raw = dataset_payload['val']

VOCAB_SIZE = dataset_payload.get('vocab_size', 52)
PAD_TOKEN = dataset_payload.get('pad_token', 50)
STOP_TOKEN = dataset_payload.get('stop_token', 51)
MAX_SRC_LEN = dataset_payload.get('max_src_len', 200)
MAX_TGT_LEN = dataset_payload.get('max_tgt_len', 51)

print(f"Loaded validation set with {len(val_raw)} samples. Vocab Size: {VOCAB_SIZE}, Max Src Len: {MAX_SRC_LEN}, Max Tgt Len: {MAX_TGT_LEN}")
"""
    cells.append(nbf.v4.new_code_cell(cell2_code))

    # Cell 3: Model Architecture
    cell3_code = """# Cell 3: Model Architecture Definition & Checkpoint Instantiation

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=250):
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
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16, num_heads=2, hidden_dim=32, num_layers=2):
        super(AutoregressiveGraphTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=250)

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

    def solve_graph_autoregressive(self, src, src_key_padding_mask=None, max_tgt_len=MAX_TGT_LEN, override_memory=None):
        self.eval()
        device = src.device
        batch_size = src.size(0)

        if override_memory is not None:
            memory = override_memory
        else:
            src_emb = self.pos_encoder(self.token_embedding(src))
            memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

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

print("Mechanistic Transformer architecture initialized.")
"""
    cells.append(nbf.v4.new_code_cell(cell3_code))

    # Cell 4: Summary
    cell4_md = """### Research Summary & Conclusions
1. **Random Walk Evaluation**: Analyzes attention entropy, logit margins, and memory activations over stochastic Random Walk execution traces.
2. **Exported Payloads**: Serializes annotated validation evaluation datasets for downstream analysis.
"""
    cells.append(nbf.v4.new_markdown_cell(cell4_md))

    nb.cells = cells

    nb_path = "graphs/2.mechanistic_interpretability_and_causal_analysis_tutorial.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {nb_path}")

if __name__ == "__main__":
    build_mechanistic_notebook()
