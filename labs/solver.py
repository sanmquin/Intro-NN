import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque

# ---------------------------------------------------------
# 1. Non-Trivial Labyrinth Generation and Visibility Processing
# ---------------------------------------------------------

def generate_labyrinth(width=10, height=10, start=(0, 0), end=(9, 9), num_dead_ends=2, num_loops=1):
    """
    Generates a sparse, non-trivial 10x10 labyrinth with choice points and dead ends.
    - Walkable paths are 0.
    - Start is 1, End is 2.
    - Path edges (visible walls) are randomly labeled between 3 and 8.
    - Hidden/non-visible walls are labeled 9.
    """
    grid = [[-1 for _ in range(width)] for _ in range(height)]

    def get_neighbors(r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                neighbors.append((nr, nc))
        return neighbors

    # 1. Find a primary path using randomized DFS
    visited = {start}
    path_found = []

    def dfs(curr, path):
        if curr == end:
            path_found.extend(path + [end])
            return True
        neighbors = get_neighbors(*curr)
        neighbors.sort(key=lambda c: abs(c[0]-end[0]) + abs(c[1]-end[1]) + random.uniform(-1.5, 1.5))
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                if dfs(n, path + [curr]):
                    return True
        return False

    dfs(start, [])

    if not path_found:
        curr_r, curr_c = start
        path_found.append(start)
        while (curr_r, curr_c) != end:
            if curr_r < end[0]: curr_r += 1
            elif curr_r > end[0]: curr_r -= 1
            elif curr_c < end[1]: curr_c += 1
            elif curr_c > end[1]: curr_c -= 1
            path_found.append((curr_r, curr_c))

    for r, c in path_found:
        grid[r][c] = 0

    # 2. Add explicit dead ends/branches
    path_cells = [c for c in path_found if c != start and c != end]
    dead_ends_carved = 0
    attempts = 0
    target_dead_ends = random.randint(1, 3) if num_dead_ends is None else num_dead_ends
    while dead_ends_carved < target_dead_ends and attempts < 100:
        attempts += 1
        branch_start = random.choice(path_cells)
        neighbors = get_neighbors(*branch_start)
        random.shuffle(neighbors)
        for n in neighbors:
            if grid[n[0]][n[1]] == -1:
                path_adj = sum(1 for wn in get_neighbors(*n) if grid[wn[0]][wn[1]] != -1)
                if path_adj == 1:
                    grid[n[0]][n[1]] = 0
                    dead_ends_carved += 1
                    curr = n
                    length = random.choice([0, 1, 2])
                    for _ in range(length):
                        candidates = []
                        for cn in get_neighbors(*curr):
                            if grid[cn[0]][cn[1]] == -1:
                                adj_walkable = sum(1 for wn in get_neighbors(*cn) if grid[wn[0]][wn[1]] != -1)
                                if adj_walkable == 1:
                                    candidates.append(cn)
                        if candidates:
                            curr = random.choice(candidates)
                            grid[curr[0]][curr[1]] = 0
                        else:
                            break
                    break

    # 3. Add controlled loops
    target_loops = random.randint(0, 2) if num_loops is None else num_loops
    loops_carved = 0
    attempts = 0
    while loops_carved < target_loops and attempts < 50:
        attempts += 1
        r = random.randint(0, height - 1)
        c = random.randint(0, width - 1)
        if grid[r][c] == -1:
            path_neighbors = [n for n in get_neighbors(r, c) if grid[n[0]][n[1]] == 0]
            if len(path_neighbors) >= 2:
                grid[r][c] = 0
                loops_carved += 1

    grid[start[0]][start[1]] = 1
    grid[end[0]][end[1]] = 2

    final_grid = [[0 for _ in range(width)] for _ in range(height)]
    for r in range(height):
        for c in range(width):
            val = grid[r][c]
            if val in (0, 1, 2):
                final_grid[r][c] = val
            else:
                is_edge = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if grid[nr][nc] in (0, 1, 2):
                                is_edge = True
                                break
                    if is_edge: break
                if is_edge:
                    final_grid[r][c] = random.randint(3, 8)
                else:
                    final_grid[r][c] = 9
    return final_grid

# ---------------------------------------------------------
# 2. BFS Shortest Path Solver (Best Case Scenario)
# ---------------------------------------------------------

def solve_bfs(grid, start=(0, 0), end=(9, 9)):
    height, width = len(grid), len(grid[0])
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        curr = path[-1]
        if curr == end: return path
        r, c = curr
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < height and 0 <= nc < width:
                if (nr, nc) not in visited and grid[nr][nc] in (0, 1, 2):
                    visited.add((nr, nc))
                    queue.append(path + [(nr, nc)])
    return None

def analyze_labyrinth(grid):
    height, width = len(grid), len(grid[0])
    walkable = []
    start, end = None, None
    for r in range(height):
        for c in range(width):
            if grid[r][c] in (0, 1, 2):
                walkable.append((r, c))
                if grid[r][c] == 1: start = (r, c)
                elif grid[r][c] == 2: end = (r, c)

    def get_neighbors(r, c):
        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < height and 0 <= nc < width: neighbors.append((nr, nc))
        return neighbors

    num_intersections = 0
    num_dead_ends = 0
    for r, c in walkable:
        walk_neigh = sum(1 for n in get_neighbors(r, c) if grid[n[0]][n[1]] in (0, 1, 2))
        if walk_neigh >= 3: num_intersections += 1
        elif walk_neigh == 1: num_dead_ends += 1

    shortest_path_len = len(solve_bfs(grid, start, end)) if solve_bfs(grid, start, end) else -1

    if num_intersections <= 2 and num_dead_ends <= 3:
        difficulty = 'Easy'
    elif num_intersections <= 4 and num_dead_ends <= 5:
        difficulty = 'Medium'
    else:
        difficulty = 'Hard'

    return {
        'walkable': len(walkable),
        'intersections': num_intersections,
        'dead_ends': num_dead_ends,
        'shortest_path': shortest_path_len,
        'difficulty': difficulty
    }

# ---------------------------------------------------------
# 3. Transition Generation & Dataset Creation
# ---------------------------------------------------------

def generate_transitions_from_shortest_path(grid, shortest_path):
    """
    Transforms a single shortest path into sequence-to-action transition steps.
    """
    transitions = []
    flat_grid = [grid[r][c] for r in range(10) for c in range(10)]

    for i in range(len(shortest_path) - 1):
        curr_r, curr_c = shortest_path[i]
        next_r, next_c = shortest_path[i+1]

        curr_idx = curr_r * 10 + curr_c
        next_idx = next_r * 10 + next_c

        transitions.append((flat_grid[:], curr_idx, next_idx))
    return transitions

class LabyrinthDataset(Dataset):
    """
    Dataset wrapping labyrinth state transitions.
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grid, curr_idx, next_idx = self.data[idx]
        return (
            torch.tensor(grid, dtype=torch.long),
            torch.tensor(curr_idx, dtype=torch.long),
            torch.tensor(next_idx, dtype=torch.long)
        )

def build_datasets(num_labyrinths=100, train_ratio=1.0, seed=42):
    """
    Generates dataset of transitions from multiple random labyrinths.
    Since we want to memorize/reconstruct known labyrinths, we don't split,
    or train/test ratio is 1.0 (train on all 100, test on all 100).
    """
    random.seed(seed)
    all_transitions = []

    success_count = 0
    attempts = 0
    while success_count < num_labyrinths and attempts < num_labyrinths * 10:
        attempts += 1
        start = (random.randint(0, 3), random.randint(0, 3))
        end = (random.randint(6, 9), random.randint(6, 9))

        grid = generate_labyrinth(width=10, height=10, start=start, end=end, num_dead_ends=2, num_loops=1)
        path = solve_bfs(grid, start=start, end=end)
        if path and len(path) > 1:
            transitions = generate_transitions_from_shortest_path(grid, path)
            all_transitions.extend(transitions)
            success_count += 1

    # Shuffle
    random.shuffle(all_transitions)
    return LabyrinthDataset(all_transitions), LabyrinthDataset(all_transitions)

# ---------------------------------------------------------
# 4. Labyrinth Transformer Model Implementation
# ---------------------------------------------------------

class LabyrinthTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=3):
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
# 5. Training and Evaluation Routine
# ---------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0

        for grids, curr_poss, targets in train_loader:
            grids, curr_poss, targets = grids.to(device), curr_poss.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(grids, curr_poss)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * grids.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += grids.size(0)

        train_loss = total_train_loss / total
        train_acc = (correct / total) * 100
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for grids, curr_poss, targets in val_loader:
                grids, curr_poss, targets = grids.to(device), curr_poss.to(device), targets.to(device)
                logits = model(grids, curr_poss)
                loss = criterion(logits, targets)

                total_val_loss += loss.item() * grids.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += grids.size(0)

        val_loss = total_val_loss / val_total
        val_acc = (val_correct / val_total) * 100
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")

    return train_losses, val_losses

# ---------------------------------------------------------
# 6. Autoregressive Labyrinth Navigation Solver
# ---------------------------------------------------------

def solve_labyrinth_autoregressive(model, grid, start, end, max_steps=40, device='cpu'):
    model.eval()
    height, width = len(grid), len(grid[0])
    curr_pos = start
    path = [start]

    flat_grid = [grid[r][c] for r in range(height) for c in range(width)]
    grid_tensor = torch.tensor([flat_grid], dtype=torch.long, device=device)

    visited = {start}

    for _ in range(max_steps):
        if curr_pos == end:
            break

        curr_idx = curr_pos[0] * 10 + curr_pos[1]
        curr_pos_tensor = torch.tensor([curr_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(grid_tensor, curr_pos_tensor).squeeze(0)
            probs = torch.softmax(logits, dim=-1)

        r, c = curr_pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if grid[nr][nc] in (0, 1, 2):
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
