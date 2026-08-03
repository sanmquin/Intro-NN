import os
import sys
# Automatically append repository root to sys.path to enable smooth imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Import solver functions from labys.solver
from labys.solver import generate_labyrinth, solve_bfs

# ---------------------------------------------------------
# 1. Model Architectures
# ---------------------------------------------------------

class LabyrinthReconstructor(nn.Module):
    """
    Reconstructs the full 10x10 labyrinth grid from a partially visible grid.
    Input shape: (batch_size, 100) with tokens 0-9
    Output shape: (batch_size, 100, 10) representing class logits for each cell.
    """
    def __init__(self, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2):
        super(LabyrinthReconstructor, self).__init__()
        self.grid_embedding = nn.Embedding(10, embed_dim)
        self.spatial_embedding = nn.Parameter(torch.randn(1, 100, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 10)  # 10 output classes representing tokens 0-9

    def forward(self, grid):
        x = self.grid_embedding(grid)
        x = x + self.spatial_embedding
        out = self.transformer(x)
        logits = self.fc_out(out)
        return logits


class LabyrinthTransformer(nn.Module):
    """
    Predicts the next navigation step in a 10x10 labyrinth.
    Can be fed either reconstructed full grids or partially visible grids.
    Input shape:
      - grid: (batch_size, 100)
      - curr_pos: (batch_size,)
    Output shape:
      - logits: (batch_size, 100)
    """
    def __init__(self, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2):
        super(LabyrinthTransformer, self).__init__()
        self.grid_embedding = nn.Embedding(10, embed_dim)
        self.pos_embedding = nn.Embedding(100, embed_dim)
        self.spatial_embedding = nn.Parameter(torch.randn(1, 100, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 100)

    def forward(self, grid, curr_pos):
        grid_emb = self.grid_embedding(grid)
        grid_emb = grid_emb + self.spatial_embedding
        pos_emb = self.pos_embedding(curr_pos).unsqueeze(1)
        x = grid_emb + pos_emb

        out = self.transformer(x)

        batch_size = grid.size(0)
        batch_indices = torch.arange(batch_size, device=grid.device)
        curr_cell_repr = out[batch_indices, curr_pos]

        logits = self.fc_out(curr_cell_repr)
        return logits

# ---------------------------------------------------------
# 2. Data Generation and Partial Visibility Logic
# ---------------------------------------------------------

def get_partial_visibility_grid(true_grid, visited_positions, start=(0, 0), end=(9, 9)):
    """
    Constructs a partially visible grid from the true grid.
    Reveals cells within Chebyshev distance 1 (3x3 neighborhood) around all visited positions.
    Start and End are always revealed. All unobserved cells are masked to 9 (hidden wall).
    """
    partial_grid = [9] * 100
    start_idx = start[0] * 10 + start[1]
    end_idx = end[0] * 10 + end[1]
    partial_grid[start_idx] = true_grid[start_idx]
    partial_grid[end_idx] = true_grid[end_idx]

    visible_cells = set()
    for vr, vc in visited_positions:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = vr + dr, vc + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    visible_cells.add((nr, nc))

    for nr, nc in visible_cells:
        idx = nr * 10 + nc
        partial_grid[idx] = true_grid[idx]

    return partial_grid


class ReconstructorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        partial_grid, true_grid = self.data[idx]
        return (
            torch.tensor(partial_grid, dtype=torch.long),
            torch.tensor(true_grid, dtype=torch.long)
        )


class SolverDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grid, curr_pos, next_pos = self.data[idx]
        return (
            torch.tensor(grid, dtype=torch.long),
            torch.tensor(curr_pos, dtype=torch.long),
            torch.tensor(next_pos, dtype=torch.long)
        )


def generate_all_datasets(num_labyrinths=310, train_ratio=0.677, seed=42):
    """
    Generates training/validation datasets of transitions from multiple random labyrinths.
    Each labyrinth is filtered to be strictly sparse and non-trivial (walkable cells between 10 and 60).
    With num_labyrinths=310 and train_ratio=0.677, we get:
      - 209 train labyrinths (to train very fast on CPU)
      - 101 test labyrinths (strictly fulfilling the requirement of testing on at least 100 labyrinths)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_labyrinths = []
    success_count = 0
    attempts = 0

    print("Generating and filtering labyrinths...")
    while success_count < num_labyrinths and attempts < num_labyrinths * 100:
        attempts += 1
        start = (random.randint(0, 2), random.randint(0, 2))
        end = (random.randint(7, 9), random.randint(7, 9))

        grid = generate_labyrinth(width=10, height=10, start=start, end=end, num_loops=random.randint(4, 7))

        # Check sparseness bounds (must have between 10 and 60 walkable cells)
        walkable_count = sum(1 for r in range(10) for c in range(10) if grid[r][c] in (0, 1, 2))
        if walkable_count < 10 or walkable_count > 60:
            continue

        path = solve_bfs(grid, start=start, end=end)
        if path and len(path) > 1:
            all_labyrinths.append((grid, path, start, end))
            success_count += 1

    print(f"Total labyrinths generated: {success_count} (out of {attempts} attempts)")

    split_idx = int(num_labyrinths * train_ratio)
    train_labyrinths = all_labyrinths[:split_idx]
    val_labyrinths = all_labyrinths[split_idx:]

    def build_samples(labyrinths_list):
        recon_data = []
        modular_solver_data = []
        monolithic_solver_data = []

        for grid, path, start, end in labyrinths_list:
            flat_true_grid = [grid[r][c] for r in range(10) for c in range(10)]

            for t in range(len(path) - 1):
                curr_pos = path[t]
                next_pos = path[t+1]

                curr_idx = curr_pos[0] * 10 + curr_pos[1]
                next_idx = next_pos[0] * 10 + next_pos[1]

                visited_positions = path[:t+1]
                partial_grid = get_partial_visibility_grid(flat_true_grid, visited_positions, start, end)

                recon_data.append((partial_grid, flat_true_grid))
                modular_solver_data.append((flat_true_grid, curr_idx, next_idx))
                monolithic_solver_data.append((partial_grid, curr_idx, next_idx))

        return recon_data, modular_solver_data, monolithic_solver_data

    train_recon, train_mod, train_mono = build_samples(train_labyrinths)
    val_recon, val_mod, val_mono = build_samples(val_labyrinths)

    # Convert to PyTorch Datasets
    train_recon_ds = ReconstructorDataset(train_recon)
    val_recon_ds = ReconstructorDataset(val_recon)

    train_mod_ds = SolverDataset(train_mod)
    val_mod_ds = SolverDataset(val_mod)

    train_mono_ds = SolverDataset(train_mono)
    val_mono_ds = SolverDataset(val_mono)

    return (
        (train_recon_ds, val_recon_ds),
        (train_mod_ds, val_mod_ds),
        (train_mono_ds, val_mono_ds),
        val_labyrinths  # To be used for full navigation inference testing
    )

# ---------------------------------------------------------
# 3. Model Training Functions
# ---------------------------------------------------------

def train_reconstructor_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    print("Training Reconstructor Model...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        total_pixels = 0

        for partial_grids, true_grids in train_loader:
            partial_grids, true_grids = partial_grids.to(device), true_grids.to(device)

            optimizer.zero_grad()
            logits = model(partial_grids) # shape (B, 100, 10)

            loss = criterion(logits.view(-1, 10), true_grids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * partial_grids.size(0)
            total_pixels += partial_grids.size(0)

        train_loss = total_train_loss / total_pixels
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_pixels = 0
        with torch.no_grad():
            for partial_grids, true_grids in val_loader:
                partial_grids, true_grids = partial_grids.to(device), true_grids.to(device)
                logits = model(partial_grids)
                loss = criterion(logits.view(-1, 10), true_grids.view(-1))

                total_val_loss += loss.item() * partial_grids.size(0)
                total_val_pixels += partial_grids.size(0)

        val_loss = total_val_loss / total_val_pixels
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Reconstructor Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


def train_solver_model(model, train_loader, val_loader, model_name="Solver", epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    print(f"Training {model_name} Model...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        total_samples = 0

        for grids, curr_poss, targets in train_loader:
            grids, curr_poss, targets = grids.to(device), curr_poss.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(grids, curr_poss)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * grids.size(0)
            total_samples += grids.size(0)

        train_loss = total_train_loss / total_samples
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            for grids, curr_poss, targets in val_loader:
                grids, curr_poss, targets = grids.to(device), curr_poss.to(device), targets.to(device)
                logits = model(grids, curr_poss)
                loss = criterion(logits, targets)

                total_val_loss += loss.item() * grids.size(0)
                total_val_samples += grids.size(0)

        val_loss = total_val_loss / total_val_samples
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{model_name} Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# ---------------------------------------------------------
# 4. Autoregressive Solvers (Inference)
# ---------------------------------------------------------

def solve_autoregressive_modular(recon_model, solver_model, grid_true, start, end, max_steps=40, device='cpu'):
    recon_model.eval()
    solver_model.eval()

    visited = {start}
    path = [start]
    curr_pos = start
    flat_true_grid = [grid_true[r][c] for r in range(10) for c in range(10)]

    step_reconstructed_accuracies = []

    for step in range(max_steps):
        if curr_pos == end:
            break

        # Get current visibility grid
        g_partial = get_partial_visibility_grid(flat_true_grid, path, start, end)
        g_partial_t = torch.tensor([g_partial], dtype=torch.long, device=device)

        with torch.no_grad():
            # 1. Reconstruction step
            recon_logits = recon_model(g_partial_t) # shape (1, 100, 10)
            recon_grid = torch.argmax(recon_logits, dim=-1) # shape (1, 100)

            # Compute reconstruction accuracy
            correct_cells = (recon_grid.squeeze(0) == torch.tensor(flat_true_grid, device=device)).sum().item()
            step_reconstructed_accuracies.append(correct_cells / 100.0)

            # 2. Solver step
            curr_pos_idx = curr_pos[0] * 10 + curr_pos[1]
            curr_pos_t = torch.tensor([curr_pos_idx], dtype=torch.long, device=device)
            logits = solver_model(recon_grid, curr_pos_t).squeeze(0)
            probs = torch.softmax(logits, dim=-1)

        # Select best valid neighbor
        r, c = curr_pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                if grid_true[nr][nc] in (0, 1, 2):
                    neighbors.append((nr, nc))

        if not neighbors:
            break

        best_neighbor = None
        best_prob = -1.0
        for nr, nc in neighbors:
            n_idx = nr * 10 + nc
            prob = probs[n_idx].item()
            if (nr, nc) in visited:
                prob *= 0.01  # Penalty for visiting already visited cells
            if prob > best_prob:
                best_prob = prob
                best_neighbor = (nr, nc)

        if best_neighbor is None:
            break

        curr_pos = best_neighbor
        path.append(curr_pos)
        visited.add(curr_pos)

    return path, step_reconstructed_accuracies


def solve_autoregressive_monolithic(mono_model, grid_true, start, end, max_steps=40, device='cpu'):
    mono_model.eval()

    visited = {start}
    path = [start]
    curr_pos = start
    flat_true_grid = [grid_true[r][c] for r in range(10) for c in range(10)]

    for step in range(max_steps):
        if curr_pos == end:
            break

        # Get current visibility grid
        g_partial = get_partial_visibility_grid(flat_true_grid, path, start, end)
        g_partial_t = torch.tensor([g_partial], dtype=torch.long, device=device)

        with torch.no_grad():
            curr_pos_idx = curr_pos[0] * 10 + curr_pos[1]
            curr_pos_t = torch.tensor([curr_pos_idx], dtype=torch.long, device=device)
            logits = mono_model(g_partial_t, curr_pos_t).squeeze(0)
            probs = torch.softmax(logits, dim=-1)

        # Select best valid neighbor
        r, c = curr_pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                if grid_true[nr][nc] in (0, 1, 2):
                    neighbors.append((nr, nc))

        if not neighbors:
            break

        best_neighbor = None
        best_prob = -1.0
        for nr, nc in neighbors:
            n_idx = nr * 10 + nc
            prob = probs[n_idx].item()
            if (nr, nc) in visited:
                prob *= 0.01  # Penalty for visiting already visited cells
            if prob > best_prob:
                best_prob = prob
                best_neighbor = (nr, nc)

        if best_neighbor is None:
            break

        curr_pos = best_neighbor
        path.append(curr_pos)
        visited.add(curr_pos)

    return path

# ---------------------------------------------------------
# 5. Core Experiment Execution Runner
# ---------------------------------------------------------

def run_experiment(epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing experiment on device: {device}")

    # 1. Generate full dataset (310 labyrinths: 209 train, 101 test)
    recon_ds, mod_ds, mono_ds, test_labyrinths = generate_all_datasets(num_labyrinths=310, train_ratio=0.677, seed=42)

    # Dataloaders - large batch size (128) to run extremely fast on CPU
    train_recon_loader = DataLoader(recon_ds[0], batch_size=128, shuffle=True)
    val_recon_loader = DataLoader(recon_ds[1], batch_size=128, shuffle=False)

    train_mod_loader = DataLoader(mod_ds[0], batch_size=128, shuffle=True)
    val_mod_loader = DataLoader(mod_ds[1], batch_size=128, shuffle=False)

    train_mono_loader = DataLoader(mono_ds[0], batch_size=128, shuffle=True)
    val_mono_loader = DataLoader(mono_ds[1], batch_size=128, shuffle=False)

    # 2. Instantiate models
    reconstructor = LabyrinthReconstructor(embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2)
    modular_solver = LabyrinthTransformer(embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2)
    monolithic_solver = LabyrinthTransformer(embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2)

    # 3. Train models
    recon_train_loss, recon_val_loss = train_reconstructor_model(reconstructor, train_recon_loader, val_recon_loader, epochs, lr, device)
    mod_train_loss, mod_val_loss = train_solver_model(modular_solver, train_mod_loader, val_mod_loader, "Modular Solver", epochs, lr, device)
    mono_train_loss, mono_val_loss = train_solver_model(monolithic_solver, train_mono_loader, val_mono_loader, "Monolithic Solver", epochs, lr, device)

    # Save checkpoints
    os.makedirs("labs", exist_ok=True)
    torch.save(reconstructor.state_dict(), "labs/reconstructor.pt")
    torch.save(modular_solver.state_dict(), "labs/modular_solver.pt")
    torch.save(monolithic_solver.state_dict(), "labs/monolithic_solver.pt")
    print("All model checkpoints saved successfully.")

    # 4. Perform comparative testing on 100 test labyrinths
    print("\nStarting comparative evaluation on 100 test labyrinths...")

    modular_successes = 0
    monolithic_successes = 0

    modular_path_efficiencies = []
    monolithic_path_efficiencies = []

    all_modular_accuracies = [] # lists of step accuracies

    # Profiling latencies
    modular_step_latencies = []
    monolithic_step_latencies = []

    for grid, opt_path, start, end in test_labyrinths:
        opt_len = len(opt_path)

        # A. Modular Architecture Navigation
        t0 = time.time()
        mod_path, mod_accs = solve_autoregressive_modular(reconstructor, modular_solver, grid, start, end, max_steps=40, device=device)
        t_elapsed = time.time() - t0

        if len(mod_path) > 1:
            modular_step_latencies.append(t_elapsed / (len(mod_path) - 1))

        if mod_path[-1] == end:
            modular_successes += 1
            modular_path_efficiencies.append(opt_len / len(mod_path))
            all_modular_accuracies.append(mod_accs)

        # B. Monolithic Architecture Navigation
        t0 = time.time()
        mono_path = solve_autoregressive_monolithic(monolithic_solver, grid, start, end, max_steps=40, device=device)
        t_elapsed = time.time() - t0

        if len(mono_path) > 1:
            monolithic_step_latencies.append(t_elapsed / (len(mono_path) - 1))

        if mono_path[-1] == end:
            monolithic_successes += 1
            monolithic_path_efficiencies.append(opt_len / len(mono_path))

    # Compile and average evaluation results
    mod_success_rate = (modular_successes / len(test_labyrinths)) * 100.0
    mono_success_rate = (monolithic_successes / len(test_labyrinths)) * 100.0

    avg_mod_efficiency = np.mean(modular_path_efficiencies) * 100.0 if modular_path_efficiencies else 0.0
    avg_mono_efficiency = np.mean(monolithic_path_efficiencies) * 100.0 if monolithic_path_efficiencies else 0.0

    avg_mod_latency = np.mean(modular_step_latencies) * 1000.0 # to ms
    avg_mono_latency = np.mean(monolithic_step_latencies) * 1000.0 # to ms

    # Trainable parameter counts
    reconstructor_params = sum(p.numel() for p in reconstructor.parameters() if p.requires_grad)
    modular_solver_params = sum(p.numel() for p in modular_solver.parameters() if p.requires_grad)
    monolithic_solver_params = sum(p.numel() for p in monolithic_solver.parameters() if p.requires_grad)

    print("\n" + "="*50)
    print("           EXPERIMENT RESULTS SUMMARY           ")
    print("="*50)
    print(f"Labyrinths Evaluated:          {len(test_labyrinths)}")
    print("-"*50)
    print(f"Modular Architecture:")
    print(f"  - Reconstructor Parameters:  {reconstructor_params:,}")
    print(f"  - Solver Parameters:         {modular_solver_params:,}")
    print(f"  - Total Parameters:          {(reconstructor_params + modular_solver_params):,}")
    print(f"  - Success Rate:              {mod_success_rate:.2f}%")
    print(f"  - Average Path Efficiency:   {avg_mod_efficiency:.2f}%")
    print(f"  - Avg Latency Per Step:      {avg_mod_latency:.3f} ms")
    print("-"*50)
    print(f"Monolithic Architecture:")
    print(f"  - Total Parameters:          {monolithic_solver_params:,}")
    print(f"  - Success Rate:              {mono_success_rate:.2f}%")
    print(f"  - Average Path Efficiency:   {avg_mono_efficiency:.2f}%")
    print(f"  - Avg Latency Per Step:      {avg_mono_latency:.3f} ms")
    print("="*50)

    # ---------------------------------------------------------
    # 6. Comparative Chart Generation
    # ---------------------------------------------------------
    os.makedirs("charts", exist_ok=True)

    # Chart A: Training Loss Trajectories
    plt.figure(figsize=(10, 5))
    plt.plot(recon_train_loss, label="Reconstructor Train Loss", color='blue', linestyle='-')
    plt.plot(recon_val_loss, label="Reconstructor Val Loss", color='blue', linestyle='--')
    plt.plot(mod_train_loss, label="Modular Solver Train Loss", color='green', linestyle='-')
    plt.plot(mod_val_loss, label="Modular Solver Val Loss", color='green', linestyle='--')
    plt.plot(mono_train_loss, label="Monolithic Solver Train Loss", color='red', linestyle='-')
    plt.plot(mono_val_loss, label="Monolithic Solver Val Loss", color='red', linestyle='--')
    plt.title("Model Training Loss Curves (10 Epochs)", fontsize=14, weight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("charts/architecture_loss_comparison.png", dpi=150)
    plt.close()

    # Chart B: Metrics Bar Chart Comparison
    labels = ['Success Rate (%)', 'Path Efficiency (%)', 'Parameters (x100k)', 'Step Latency (ms)']
    modular_metrics = [mod_success_rate, avg_mod_efficiency, (reconstructor_params + modular_solver_params)/100000.0, avg_mod_latency]
    monolithic_metrics = [mono_success_rate, avg_mono_efficiency, monolithic_solver_params/100000.0, avg_mono_latency]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, modular_metrics, width, label='Modular Architecture', color='teal')
    rects2 = ax.bar(x + width/2, monolithic_metrics, width, label='Monolithic Architecture', color='crimson')

    ax.set_ylabel('Metric Values', fontsize=12)
    ax.set_title('Modular vs Monolithic Performance & Cost Comparison', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig("charts/architecture_cost_metrics.png", dpi=150)
    plt.close()

    # Chart C: Reconstruction Accuracy Over Step-by-Step Navigation
    # Since different labyrinths have different step counts, we align them by step index
    max_steps_tested = max(len(acc) for acc in all_modular_accuracies) if all_modular_accuracies else 0
    accuracy_by_step = [[] for _ in range(max_steps_tested)]
    for acc_list in all_modular_accuracies:
        for idx, val in enumerate(acc_list):
            accuracy_by_step[idx].append(val)

    avg_accuracy_by_step = [np.mean(steps_accs)*100.0 for steps_accs in accuracy_by_step if len(steps_accs) > 0]

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(avg_accuracy_by_step) + 1), avg_accuracy_by_step, marker='o', color='purple', linewidth=2.5)
    plt.title("Modular Reconstruction Accuracy Progress During Exploration", fontsize=13, weight='bold')
    plt.xlabel("Navigation Step Index", fontsize=11)
    plt.ylabel("Maze Reconstruction Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("charts/reconstruction_accuracy_trajectory.png", dpi=150)
    plt.close()

    print("Comparative charts generated and saved successfully under 'charts/' directory.")


if __name__ == "__main__":
    # If executed directly, run the complete experiment
    run_experiment(epochs=10, lr=1e-3)
