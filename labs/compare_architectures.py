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
from labys.solver import generate_labyrinth, solve_bfs, analyze_labyrinth

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
# 2. Partial Visibility & Datasets
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

# ---------------------------------------------------------
# 3. Autoregressive Navigation Solvers
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

        g_partial = get_partial_visibility_grid(flat_true_grid, path, start, end)
        g_partial_t = torch.tensor([g_partial], dtype=torch.long, device=device)

        with torch.no_grad():
            # 1. Reconstruction step
            recon_logits = recon_model(g_partial_t)
            recon_grid = torch.argmax(recon_logits, dim=-1)

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
                prob *= 0.01
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
                prob *= 0.01
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
# 4. Core Experiment Runner
# ---------------------------------------------------------

def run_experiment(epochs=40, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing experiment on device: {device}")

    # Generate 100 non-trivial labyrinths with explicit choice points and dead ends
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    labyrinths = []
    print("--------------------------------------------------")
    print("STEP 1: Generating 100 Labyrinths for Associative Memorization")
    print("--------------------------------------------------")
    for i in range(100):
        start = (random.randint(0, 2), random.randint(0, 2))
        end = (random.randint(7, 9), random.randint(7, 9))
        grid = generate_labyrinth(width=10, height=10, start=start, end=end, num_dead_ends=2, num_loops=1)
        path = solve_bfs(grid, start=start, end=end)
        while not path or len(path) <= 1:
            grid = generate_labyrinth(width=10, height=10, start=start, end=end, num_dead_ends=2, num_loops=1)
            path = solve_bfs(grid, start=start, end=end)
        stats = analyze_labyrinth(grid)
        labyrinths.append((grid, path, start, end, stats['difficulty']))

    # Print difficulty counts
    diff_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    for _, _, _, _, diff in labyrinths:
        diff_counts[diff] += 1
    print(f"Labyrinth Generation Complete! Difficulty breakdown:")
    for k, v in diff_counts.items():
        print(f"  - {k} Difficulty: {v} mazes")
    print("--------------------------------------------------\n")

    # Build datasets
    recon_data = []
    mod_data = []
    mono_data = []

    for grid, path, start, end, diff in labyrinths:
        flat_true_grid = [grid[r][c] for r in range(10) for c in range(10)]
        for t in range(len(path) - 1):
            curr_pos = path[t]
            next_pos = path[t+1]
            curr_idx = curr_pos[0] * 10 + curr_pos[1]
            next_idx = next_pos[0] * 10 + next_pos[1]

            visited = path[:t+1]
            g_partial = get_partial_visibility_grid(flat_true_grid, visited, start, end)

            recon_data.append((g_partial, flat_true_grid))
            mod_data.append((flat_true_grid, curr_idx, next_idx))
            mono_data.append((g_partial, curr_idx, next_idx))

    train_recon_ds = ReconstructorDataset(recon_data)
    train_mod_ds = SolverDataset(mod_data)
    train_mono_ds = SolverDataset(mono_data)

    train_recon_loader = DataLoader(train_recon_ds, batch_size=128, shuffle=True)
    train_mod_loader = DataLoader(train_mod_ds, batch_size=128, shuffle=True)
    train_mono_loader = DataLoader(train_mono_ds, batch_size=128, shuffle=True)

    # Instantiate models
    reconstructor = LabyrinthReconstructor(embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2)
    modular_solver = LabyrinthTransformer(embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2)
    monolithic_solver = LabyrinthTransformer(embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2)

    # Train Reconstructor
    print("--------------------------------------------------")
    print("STEP 2: Training Reconstructor (CA3 Attractor Network)")
    print("--------------------------------------------------")
    optimizer = optim.AdamW(reconstructor.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    recon_train_loss = []
    for epoch in range(1, epochs + 1):
        reconstructor.train()
        total_loss = 0
        for p, t in train_recon_loader:
            p, t = p.to(device), t.to(device)
            optimizer.zero_grad()
            logits = reconstructor(p)
            loss = criterion(logits.view(-1, 10), t.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * p.size(0)
        epoch_loss = total_loss / len(train_recon_ds)
        recon_train_loss.append(epoch_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Reconstructor] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f}")

    # Train Modular Solver
    print("\n--------------------------------------------------")
    print("STEP 3: Training Modular Solver (CA1 Directional Planning)")
    print("--------------------------------------------------")
    optimizer = optim.AdamW(modular_solver.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    mod_train_loss = []
    for epoch in range(1, epochs + 1):
        modular_solver.train()
        total_loss = 0
        for g, p, t in train_mod_loader:
            g, p, t = g.to(device), p.to(device), t.to(device)
            optimizer.zero_grad()
            logits = modular_solver(g, p)
            loss = criterion(logits, t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * g.size(0)
        epoch_loss = total_loss / len(train_mod_ds)
        mod_train_loss.append(epoch_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Modular Solver] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f}")

    # Train Monolithic Solver
    print("\n--------------------------------------------------")
    print("STEP 4: Training Monolithic Solver (Direct Sensorimotor Map)")
    print("--------------------------------------------------")
    optimizer = optim.AdamW(monolithic_solver.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    mono_train_loss = []
    for epoch in range(1, epochs + 1):
        monolithic_solver.train()
        total_loss = 0
        for g, p, t in train_mono_loader:
            g, p, t = g.to(device), p.to(device), t.to(device)
            optimizer.zero_grad()
            logits = monolithic_solver(g, p)
            loss = criterion(logits, t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * g.size(0)
        epoch_loss = total_loss / len(train_mono_ds)
        mono_train_loss.append(epoch_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Monolithic Solver] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f}")

    # Save Checkpoints
    os.makedirs("labs", exist_ok=True)
    torch.save(reconstructor.state_dict(), "labs/reconstructor.pt")
    torch.save(modular_solver.state_dict(), "labs/modular_solver.pt")
    torch.save(monolithic_solver.state_dict(), "labs/monolithic_solver.pt")
    print("\nAll model checkpoints successfully saved in 'labs/'.")

    # Evaluation
    print("\n--------------------------------------------------")
    print("STEP 5: Executing Comparative Evaluation Under Partial Observability")
    print("--------------------------------------------------")
    print("Testing started on all 100 memorized environments step-by-step...\n")

    results = {'Modular': [], 'Monolithic': []}
    all_modular_accuracies = []

    for idx, (grid, opt_path, start, end, diff) in enumerate(labyrinths):
        opt_len = len(opt_path)

        # 1. Modular Architecture
        mod_path, mod_accs = solve_autoregressive_modular(reconstructor, modular_solver, grid, start, end, max_steps=40, device=device)
        mod_success = 1 if mod_path[-1] == end else 0
        mod_len = len(mod_path)
        mod_eff = opt_len / mod_len if mod_success else 0.0
        mod_missteps = sum(1 for c in mod_path if c not in opt_path)
        mod_backtracks = mod_len - len(set(mod_path))

        results['Modular'].append((mod_success, mod_eff, mod_missteps, mod_backtracks, diff))
        if mod_success:
            all_modular_accuracies.append(mod_accs)

        # 2. Monolithic Architecture
        mono_path = solve_autoregressive_monolithic(monolithic_solver, grid, start, end, max_steps=40, device=device)
        mono_success = 1 if mono_path[-1] == end else 0
        mono_len = len(mono_path)
        mono_eff = opt_len / mono_len if mono_success else 0.0
        mono_missteps = sum(1 for c in mono_path if c not in opt_path)
        mono_backtracks = mono_len - len(set(mono_path))

        results['Monolithic'].append((mono_success, mono_eff, mono_missteps, mono_backtracks, diff))

        # Printing individual exploration logs for visibility
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Maze #{idx+1:03d} ({diff:6s}) | Opt Path: {opt_len:2d} | Modular: Succ={mod_success}, Steps={mod_len:2d}, Backtracks={mod_backtracks:2d} | Monolithic: Succ={mono_success}, Steps={mono_len:2d}, Backtracks={mono_backtracks:2d}")

    # Print results summary
    print("\n" + "="*65)
    print("               EVALUATION METRICS SUMMARY & DIFFICULTY ANALYSIS")
    print("="*65)
    for arch in ['Modular', 'Monolithic']:
        successes, effs, missteps, backtracks, _ = zip(*results[arch])
        print(f"\n{arch} Architecture (Global Metrics across all 100 mazes):")
        print(f"  - Success Rate:              {np.mean(successes)*100:.2f}%")
        print(f"  - Average Path Efficiency:   {np.mean(effs)*100:.2f}%")
        print(f"  - Avg Missteps Per Run:      {np.mean(missteps):.2f}")
        print(f"  - Avg Backtracks Per Run:    {np.mean(backtracks):.2f}")

        for d in ['Easy', 'Medium', 'Hard']:
            d_sub = [results[arch][i] for i, x in enumerate(results[arch]) if x[4] == d]
            if d_sub:
                d_succs, d_effs, d_miss, d_back, _ = zip(*d_sub)
                print(f"  [{d} Difficulty ({len(d_sub)} mazes)]:")
                print(f"    - Success Rate:            {np.mean(d_succs)*100:.2f}%")
                print(f"    - Average Path Efficiency: {np.mean(d_effs)*100:.2f}%")
                print(f"    - Avg Missteps Per Run:    {np.mean(d_miss):.2f}")
                print(f"    - Avg Backtracks Per Run:  {np.mean(d_back):.2f}")
    print("="*65 + "\n")

    # Save updated comparison charts
    os.makedirs("charts", exist_ok=True)

    # 1. Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(recon_train_loss, label="Reconstructor Train Loss", color='blue', linewidth=2)
    plt.plot(mod_train_loss, label="Modular Solver Train Loss", color='green', linewidth=2)
    plt.plot(mono_train_loss, label="Monolithic Solver Train Loss", color='red', linewidth=2)
    plt.title("Labyrinth Model Training Loss Curves (40 Epochs)", fontsize=14, weight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("charts/architecture_loss_comparison.png", dpi=150)
    plt.close()

    # 2. Global Performance Metrics Chart
    mod_success_rate = np.mean([x[0] for x in results['Modular']]) * 100
    mono_success_rate = np.mean([x[0] for x in results['Monolithic']]) * 100
    avg_mod_efficiency = np.mean([x[1] for x in results['Modular']]) * 100
    avg_mono_efficiency = np.mean([x[1] for x in results['Monolithic']]) * 100
    avg_mod_missteps = np.mean([x[2] for x in results['Modular']])
    avg_mono_missteps = np.mean([x[2] for x in results['Monolithic']])
    avg_mod_backtracks = np.mean([x[3] for x in results['Modular']])
    avg_mono_backtracks = np.mean([x[3] for x in results['Monolithic']])

    labels = ['Success (%)', 'Path Efficiency (%)', 'Avg Missteps', 'Avg Backtracks']
    mod_vals = [mod_success_rate, avg_mod_efficiency, avg_mod_missteps, avg_mod_backtracks]
    mono_vals = [mono_success_rate, avg_mono_efficiency, avg_mono_missteps, avg_mono_backtracks]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, mod_vals, width, label='Modular Architecture', color='teal')
    rects2 = ax.bar(x + width/2, mono_vals, width, label='Monolithic Architecture', color='crimson')

    ax.set_ylabel('Metric Values', fontsize=12)
    ax.set_title('Modular vs Monolithic Performance under Partial Observability', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig("charts/architecture_cost_metrics.png", dpi=150)
    plt.close()

    # 3. Difficulty-Wise Success Rate Comparison
    difficulties = ['Easy', 'Medium', 'Hard']
    mod_diff_success = []
    mono_diff_success = []
    for d in difficulties:
        m_sub = [x[0] for x in results['Modular'] if x[4] == d]
        mo_sub = [x[0] for x in results['Monolithic'] if x[4] == d]
        mod_diff_success.append(np.mean(m_sub)*100 if m_sub else 0)
        mono_diff_success.append(np.mean(mo_sub)*100 if mo_sub else 0)

    x = np.arange(len(difficulties))
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, mod_diff_success, width, label='Modular Architecture', color='teal')
    rects2 = ax.bar(x + width/2, mono_diff_success, width, label='Monolithic Architecture', color='crimson')
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Comparison by Labyrinth Difficulty', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig("charts/difficulty_success_comparison.png", dpi=150)
    plt.close()

    # 4. Difficulty-Wise Path Efficiency Comparison
    mod_diff_eff = []
    mono_diff_eff = []
    for d in difficulties:
        m_sub = [x[1] for x in results['Modular'] if x[4] == d]
        mo_sub = [x[1] for x in results['Monolithic'] if x[4] == d]
        mod_diff_eff.append(np.mean(m_sub)*100 if m_sub else 0)
        mono_diff_eff.append(np.mean(mo_sub)*100 if mo_sub else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, mod_diff_eff, width, label='Modular Architecture', color='teal')
    rects2 = ax.bar(x + width/2, mono_diff_eff, width, label='Monolithic Architecture', color='crimson')
    ax.set_ylabel('Average Path Efficiency (%)', fontsize=12)
    ax.set_title('Path Efficiency Comparison by Labyrinth Difficulty', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig("charts/difficulty_efficiency_comparison.png", dpi=150)
    plt.close()

    # 5. Reconstruction Accuracy Over Step Index
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

    print("All updated comparative charts successfully generated and saved under 'charts/'.")

if __name__ == "__main__":
    run_experiment(epochs=40, lr=1e-3)
