import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import math
import numpy as np
import random
import os

# Reduce threads to speed up PyTorch training on CPU
torch.set_num_threads(1)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()

SEQ_LEN = 5
VOCAB_SIZE = 10
D_MODEL = 32
N_HEADS = 2
D_FF = 64
N_LAYERS = 2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x)
        x = x + attn_out
        norm_x2 = self.norm2(x)
        ff_out = self.ff(norm_x2)
        x = x + ff_out
        return x, attn_weights

class TransformerSorter(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len=seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        out = self.embedding(x)
        out = self.pe(out)
        all_attn_weights = []
        for layer in self.layers:
            out, attn_weights = layer(out)
            all_attn_weights.append(attn_weights)
        out = self.norm(out)
        logits = self.fc_out(out)
        return logits, all_attn_weights

# Generate data
def create_dataset(num_samples):
    inputs = []
    targets = []
    for _ in range(num_samples):
        inp = [random.randint(0, 9) for _ in range(5)]
        tar = sorted(inp)
        inputs.append(inp)
        targets.append(tar)
    return torch.tensor(inputs), torch.tensor(targets)

train_inputs, train_targets = create_dataset(30000)
val_inputs, val_targets = create_dataset(5000)

train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=256, shuffle=False)

model = TransformerSorter(VOCAB_SIZE, SEQ_LEN, D_MODEL, N_HEADS, D_FF, N_LAYERS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

for epoch in range(1, 16):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

    # Val check
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            logits, _ = model(inputs)
            preds = torch.argmax(logits, dim=-1)
            correct_tokens += (preds == targets).sum().item()
            total_tokens += targets.numel()
            correct_sequences += (preds == targets).all(dim=-1).sum().item()

    token_acc = correct_tokens / total_tokens
    seq_acc = correct_sequences / len(val_loader.dataset)
    print(f"Epoch {epoch:02d} | Train Loss: {total_loss / len(train_loader.dataset):.4f} | Val Token Acc: {token_acc*100:.2f}% | Val Seq Acc: {seq_acc*100:.2f}%")
    if seq_acc > 0.999:
        print("Convergence achieved!")
        break

# Export weights to JSON
# Make sure directory exists
os.makedirs("web/src", exist_ok=True)

# Helper to convert tensor to nested lists
def t2l(tensor):
    return tensor.detach().cpu().numpy().tolist()

weights = {
    "embedding": t2l(model.embedding.weight),
    "pe": t2l(model.pe.pe), # Positional encodings (5, 32)
    "layers": []
}

for layer in model.layers:
    layer_weights = {
        "norm1": {
            "weight": t2l(layer.norm1.weight),
            "bias": t2l(layer.norm1.bias)
        },
        "attn": {
            "W_q": t2l(layer.attn.W_q.weight),
            "W_k": t2l(layer.attn.W_k.weight),
            "W_v": t2l(layer.attn.W_v.weight),
            "W_o": t2l(layer.attn.W_o.weight)
        },
        "norm2": {
            "weight": t2l(layer.norm2.weight),
            "bias": t2l(layer.norm2.bias)
        },
        "ff": {
            "fc1": {
                "weight": t2l(layer.ff.fc1.weight),
                "bias": t2l(layer.ff.fc1.bias)
            },
            "fc2": {
                "weight": t2l(layer.ff.fc2.weight),
                "bias": t2l(layer.ff.fc2.bias)
            }
        }
    }
    weights["layers"].append(layer_weights)

weights["norm"] = {
    "weight": t2l(model.norm.weight),
    "bias": t2l(model.norm.bias)
}
weights["fc_out"] = {
    "weight": t2l(model.fc_out.weight),
    "bias": t2l(model.fc_out.bias)
}

with open("web/src/model_weights.json", "w") as f:
    json.dump(weights, f, indent=2)

print("Export completed successfully!")
