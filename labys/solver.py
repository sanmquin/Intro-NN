import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque

# ---------------------------------------------------------
# 1. Labyrinth Generation and Visibility Processing
# ---------------------------------------------------------

def generate_labyrinth(width=10, height=10, start=(0, 0), end=(9, 9), num_extra_paths=25):
    """
    Generates a 10x10 labyrinth.
    - Walkable paths are 0.
    - Start is 1, End is 2.
    - Path edges (visible walls) are labeled with random numbers between 3 and 8 inclusive.
    - Hidden/non-visible walls are labeled 9.

    The generation guarantees at least one valid path from start to end,
    and carves extra paths to create multiple paths/loops.
    """
    grid = [[-1 for _ in range(width)] for _ in range(height)]

    # Ensure start and end are walkable paths initially
    grid[start[0]][start[1]] = 0
    grid[end[0]][end[1]] = 0

    def get_neighbors(r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                neighbors.append((nr, nc))
        return neighbors

    # Simple DFS to create a spanning tree maze
    visited = {start}
    stack = [start]
    while stack:
        curr = stack[-1]
        unvisited = [n for n in get_neighbors(*curr) if n not in visited]
        if unvisited:
            next_cell = random.choice(unvisited)
            grid[next_cell[0]][next_cell[1]] = 0
            visited.add(next_cell)
            stack.append(next_cell)
        else:
            stack.pop()

    grid[start[0]][start[1]] = 0
    grid[end[0]][end[1]] = 0

    # Add extra paths to ensure multiple paths (loops)
    # We increase this to guarantee high density of multiple paths
    for _ in range(num_extra_paths):
        wall_cells = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == -1]
        if wall_cells:
            valid_walls = []
            for wr, wc in wall_cells:
                has_path_neighbor = any(grid[nr][nc] == 0 for nr, nc in get_neighbors(wr, wc))
                if has_path_neighbor:
                    valid_walls.append((wr, wc))
            if valid_walls:
                to_carve = random.choice(valid_walls)
                grid[to_carve[0]][to_carve[1]] = 0

    grid[start[0]][start[1]] = 1
    grid[end[0]][end[1]] = 2

    # Compute visibility:
    # Any wall cell adjacent to a path cell (value 0, 1, or 2) in orth/diag direction is an edge.
    # It gets a random integer between 3 and 8.
    # Inner wall cells without any path adjacency are labeled 9.
    final_grid = [[0 for _ in range(width)] for _ in range(height)]
    for r in range(height):
        for c in range(width):
            val = grid[r][c]
            if val in (0, 1, 2):
                final_grid[r][c] = val
            else:
                # Check for path-adjacency (orthogonal or diagonal)
                is_edge = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if grid[nr][nc] in (0, 1, 2):
                                is_edge = True
                                break
                    if is_edge:
                        break

                if is_edge:
                    final_grid[r][c] = random.randint(3, 8)
                else:
                    final_grid[r][c] = 9

    return final_grid

# ---------------------------------------------------------
# 2. BFS Shortest Path Solver (Best Case Scenario)
# ---------------------------------------------------------

def solve_bfs(grid, start=(0, 0), end=(9, 9)):
    """
    Computes the absolute shortest path from start to end using Breadth-First Search (BFS).
    Walkable cells are those with grid values in {0, 1, 2}.
    Returns a list of coordinates representing the path, or None if no path exists.
    """
    height = len(grid)
    width = len(grid[0])
    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        curr = path[-1]
        if curr == end:
            return path

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if (nr, nc) not in visited and grid[nr][nc] in (0, 1, 2):
                    visited.add((nr, nc))
                    queue.append(path + [(nr, nc)])
    return None

# ---------------------------------------------------------
# 3. Transition Generation & Dataset Creation
# ---------------------------------------------------------

def generate_transitions_from_shortest_path(grid, shortest_path):
    """
    Transforms a single shortest path into sequence-to-action transition steps.
    Each step provides:
      - The static 100-token representation of the labyrinth.
      - The current position index (0-99).
      - The target next step position index (0-99).
    """
    transitions = []
    # Flat labyrinth grid
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
    Input: Flat grid (100 tokens, vocabulary size 10) + Current position index.
    Target: Next position index (0-99).
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

def build_datasets(num_labyrinths=200, train_ratio=0.8, seed=42):
    """
    Generates dataset of transitions from multiple random labyrinths.
    """
    random.seed(seed)
    all_transitions = []

    success_count = 0
    attempts = 0
    while success_count < num_labyrinths and attempts < num_labyrinths * 10:
        attempts += 1
        # Random start/end configurations on the boundaries or corners
        # Let's keep it robust and interesting by varying start/end
        start = (random.randint(0, 3), random.randint(0, 3))
        end = (random.randint(6, 9), random.randint(6, 9))

        grid = generate_labyrinth(width=10, height=10, start=start, end=end, num_extra_paths=random.randint(15, 30))
        path = solve_bfs(grid, start=start, end=end)
        if path and len(path) > 1:
            transitions = generate_transitions_from_shortest_path(grid, path)
            all_transitions.extend(transitions)
            success_count += 1

    # Shuffle and split
    random.shuffle(all_transitions)
    split_idx = int(len(all_transitions) * train_ratio)

    train_data = all_transitions[:split_idx]
    val_data = all_transitions[split_idx:]

    return LabyrinthDataset(train_data), LabyrinthDataset(val_data)

# ---------------------------------------------------------
# 4. Labyrinth Transformer Model Implementation
# ---------------------------------------------------------

class LabyrinthTransformer(nn.Module):
    """
    A Transformer architecture to navigate a 10x10 labyrinth.
    Takes a 100-token representation of the labyrinth grid (vocabulary 0..9)
    and a token representing the current position.

    Embeds both grid and position, adds learned spatial coordinate embeddings,
    processes with a Transformer Encoder, and uses a cross-attention or direct pooling
    to map to a 100-class action projection representing the next step choice.
    """
    def __init__(self, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=3):
        super(LabyrinthTransformer, self).__init__()
        # Vocabulary: 0-9 (path, start, end, edges 3-8, hidden walls 9)
        self.grid_embedding = nn.Embedding(10, embed_dim)

        # We also embed the current position index (0-99)
        self.pos_embedding = nn.Embedding(100, embed_dim)

        # Spatial coordinate embedding for the 100 positions in the grid
        self.spatial_embedding = nn.Parameter(torch.randn(1, 100, embed_dim))

        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head projecting sequence-level state to 100 target grid cells
        self.fc_out = nn.Linear(embed_dim, 100)

    def forward(self, grid, curr_pos):
        # grid shape: [Batch, 100], curr_pos shape: [Batch]
        grid_emb = self.grid_embedding(grid)  # [Batch, 100, embed_dim]
        grid_emb = grid_emb + self.spatial_embedding  # Add spatial layout context

        # Inject the current position by adding its embedding to the grid embedding
        # or concatenating. Let's do positional additive injection for elegance.
        pos_emb = self.pos_embedding(curr_pos).unsqueeze(1)  # [Batch, 1, embed_dim]

        # Broadly combine current position context with grid layout
        x = grid_emb + pos_emb  # Broadly broadcast current position context to all cells

        # Process via transformer
        out = self.transformer(x)  # [Batch, 100, embed_dim]

        # We can extract the representation of the cell corresponding to the current position
        # to predict the next step.
        batch_size = grid.size(0)
        batch_indices = torch.arange(batch_size, device=grid.device)
        curr_cell_repr = out[batch_indices, curr_pos]  # [Batch, embed_dim]

        logits = self.fc_out(curr_cell_repr)  # [Batch, 100]
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
    """
    Navigates the labyrinth step-by-step from start to end using the trained transformer.
    At each step:
      - Predict the next step logit weights for all 100 positions.
      - Filter predictions to enforce valid spatial moves (up, down, left, right to a walkable tile).
      - Step to the highly-weighted neighbor.
      - Return the actual navigated path.
    """
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
            logits = model(grid_tensor, curr_pos_tensor).squeeze(0)  # [100]
            probs = torch.softmax(logits, dim=-1)

        # Filter neighbors
        r, c = curr_pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if grid[nr][nc] in (0, 1, 2):  # Walkable
                    neighbors.append((nr, nc))

        if not neighbors:
            break

        # Score neighbors using model probabilities
        best_neighbor = None
        best_prob = -1.0

        for nr, nc in neighbors:
            n_idx = nr * 10 + nc
            prob = probs[n_idx].item()
            # To prevent simple loops, we slightly penalize already visited neighbors
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
