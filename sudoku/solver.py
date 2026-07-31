import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 4x4 Sudoku Programmatic Generation & Solvability
# ---------------------------------------------------------

def is_valid_sudoku(grid):
    """Checks if a 4x4 Sudoku grid violates rules."""
    for r in range(4):
        vals = [grid[r][c] for c in range(4) if grid[r][c] != 0]
        if len(vals) != len(set(vals)):
            return False
    for c in range(4):
        vals = [grid[r][c] for r in range(4) if grid[r][c] != 0]
        if len(vals) != len(set(vals)):
            return False
    for b in range(4):
        vals = []
        br, bc = (b // 2) * 2, (b % 2) * 2
        for r in range(br, br + 2):
            for c in range(bc, bc + 2):
                if grid[r][c] != 0:
                    vals.append(grid[r][c])
        if len(vals) != len(set(vals)):
            return False
    return True

def generate_all_solved_grids():
    """Generates all 288 valid solved 4x4 Sudoku grids."""
    solved_grids = []
    grid = [[0]*4 for _ in range(4)]
    def backtrack(r, c):
        if r == 4:
            solved_grids.append([row[:] for row in grid])
            return
        next_r = r + (c + 1) // 4
        next_c = (c + 1) % 4
        for val in range(1, 5):
            grid[r][c] = val
            if is_valid_sudoku(grid):
                backtrack(next_r, next_c)
            grid[r][c] = 0
    backtrack(0, 0)
    return solved_grids

def get_candidates(grid, r, c):
    """Returns the set of valid candidates for cell (r, c) under Sudoku rules."""
    if grid[r][c] != 0:
        return set()
    vals = set(range(1, 5))
    vals -= set(grid[r])
    vals -= set(grid[i][c] for i in range(4))
    br, bc = (r // 2) * 2, (c // 2) * 2
    for i in range(br, br + 2):
        for j in range(bc, bc + 2):
            vals.discard(grid[i][j])
    return vals

def find_naked_singles(grid):
    """Finds all empty cells that have exactly one candidate (naked single)."""
    singles = []
    for r in range(4):
        for c in range(4):
            if grid[r][c] == 0:
                cand = get_candidates(grid, r, c)
                if len(cand) == 1:
                    singles.append((r, c, list(cand)[0]))
    return singles

def generate_greedy_puzzle_path(solved_grid):
    """
    Generates a list of partial grids from a puzzle state to the solved grid.
    Each intermediate puzzle grid in the path has at least one naked single,
    meaning we can greedily solve the puzzle step-by-step.

    Returns:
        List[List[List[int]]]: Path of 4x4 grids starting from the partial puzzle to the solved grid.
    """
    current = [row[:] for row in solved_grid]
    filled_cells = [(r, c) for r in range(4) for c in range(4)]
    random.shuffle(filled_cells)

    # Try to remove elements while keeping the cell a naked single in the resulting grid.
    for r, c in filled_cells:
        val = current[r][c]
        current[r][c] = 0
        cand = get_candidates(current, r, c)
        if len(cand) == 1 and list(cand)[0] == val:
            # It's a valid naked single cell removal
            pass
        else:
            # Undo
            current[r][c] = val

    # Now generate the step-by-step solved path from 'current' to 'solved_grid'.
    path = [[row[:] for row in current]]
    temp_grid = [row[:] for row in current]

    while True:
        singles = find_naked_singles(temp_grid)
        if not singles:
            break
        # Take the first single and fill it
        r, c, val = singles[0]
        temp_grid[r][c] = val
        path.append([row[:] for row in temp_grid])

    return path

# ---------------------------------------------------------
# 2. PyTorch Dataset & DataLoader Setup
# ---------------------------------------------------------

class SudokuDataset(Dataset):
    """
    Each sample represents a state transition in a greedily-solvable Sudoku path.
    Input: Flat 16-element sequence of token IDs (0 for empty, 1-4 for values).
    Target: Flat 16-element sequence of class labels (0 for empty or unmasked, 1-4 for true values).
    Mask: Flat 16-element sequence indicating which target values to compute loss on.
          We train the model to:
          - Preserve existing non-zero values (mask = 1).
          - Predict the deterministic naked single values (mask = 1).
          - Other empty cells (not naked singles yet) have mask = 0.
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, target, mask = self.data[idx]
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32)
        )

def build_datasets(solved_grids, train_ratio=0.8, seed=42, augment_factor=5):
    """
    Splits the 288 solved grids into train/test to prevent leakage,
    then constructs the transition pairs (Input, Target, Mask) for both sets.
    Applies data augmentation by generating multiple random paths per solved grid.
    """
    random.seed(seed)
    # Shuffle solved grids
    grids = solved_grids[:]
    random.shuffle(grids)

    split_idx = int(len(grids) * train_ratio)
    train_grids = grids[:split_idx]
    test_grids = grids[split_idx:]

    def extract_transitions(grid_list, is_train=True):
        transitions = set()  # Use set to avoid duplicates
        for g in grid_list:
            # Generate multiple paths to augment dataset
            factor = augment_factor if is_train else 2
            for _ in range(factor):
                path = generate_greedy_puzzle_path(g)
                for i in range(len(path) - 1):
                    curr_grid = path[i]
                    next_grid = path[i+1]

                    # Flat representations
                    flat_curr = tuple(curr_grid[r][c] for r in range(4) for c in range(4))
                    flat_next = tuple(next_grid[r][c] for r in range(4) for c in range(4))

                    # Mask: 1 for already filled cells AND the cell that changes.
                    mask = [0] * 16
                    for idx in range(16):
                        if flat_curr[idx] != 0:
                            mask[idx] = 1 # Keep filled cells consistent
                        elif flat_next[idx] != 0:
                            mask[idx] = 1 # Target single

                    transitions.add((flat_curr, flat_next, tuple(mask)))
        return [([list(inp), list(tgt), list(msk)]) for inp, tgt, msk in transitions]

    train_data = extract_transitions(train_grids, is_train=True)
    test_data = extract_transitions(test_grids, is_train=False)

    return SudokuDataset(train_data), SudokuDataset(test_data), test_grids

# ---------------------------------------------------------
# 3. Transformer Model Implementation
# ---------------------------------------------------------

class SudokuTransformer(nn.Module):
    """
    A Transformer architecture to solve 4x4 Sudoku grids.
    It takes an input of sequence length 16, token IDs 0..4,
    adds learned positional encodings, processes via PyTorch's
    TransformerEncoder, and maps back to logits for each cell (5 classes: 0..4).
    """
    def __init__(self, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2):
        super(SudokuTransformer, self).__init__()
        # Vocabulary size is 5: 0 for empty, 1, 2, 3, 4 for digits.
        self.embedding = nn.Embedding(5, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 5) # logits for 5 classes (0, 1, 2, 3, 4)

    def forward(self, x):
        # x shape: [Batch, 16]
        x_emb = self.embedding(x) # [Batch, 16, embed_dim]
        x_emb = x_emb + self.pos_embedding # Broadcast along batch dimension

        out = self.transformer(x_emb) # [Batch, 16, embed_dim]
        logits = self.fc_out(out) # [Batch, 16, 5]
        return logits

# ---------------------------------------------------------
# 4. Training and Evaluation Routine
# ---------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')

    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        total_train_tokens = 0

        for inputs, targets, masks in train_loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(inputs) # [Batch, 16, 5]

            # Reshape for cross entropy
            logits_flat = logits.view(-1, 5)
            targets_flat = targets.view(-1)
            masks_flat = masks.view(-1)

            loss = criterion(logits_flat, targets_flat)
            # Apply masking to ignore non-target cells
            masked_loss = (loss * masks_flat).sum()
            num_tokens = masks_flat.sum()

            if num_tokens > 0:
                loss_step = masked_loss / num_tokens
                loss_step.backward()
                optimizer.step()

                total_train_loss += masked_loss.item()
                total_train_tokens += num_tokens.item()

        train_epoch_loss = total_train_loss / max(total_train_tokens, 1)
        train_losses.append(train_epoch_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_tokens = 0
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
                logits = model(inputs)
                logits_flat = logits.view(-1, 5)
                targets_flat = targets.view(-1)
                masks_flat = masks.view(-1)

                loss = criterion(logits_flat, targets_flat)
                masked_loss = (loss * masks_flat).sum()
                num_tokens = masks_flat.sum()

                total_val_loss += masked_loss.item()
                total_val_tokens += num_tokens.item()

        val_epoch_loss = total_val_loss / max(total_val_tokens, 1)
        val_losses.append(val_epoch_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

    return train_losses, val_losses

# ---------------------------------------------------------
# 5. Autoregressive Puzzle Solver (Evaluation)
# ---------------------------------------------------------

def solve_puzzle_autoregressive(model, start_grid, device='cpu'):
    """
    Solves a 4x4 Sudoku puzzle step-by-step using autoregressive greedy decoding.
    At each step:
      - We pass the current 16-token sequence to the model.
      - We find the empty cell where the model's confidence for any non-zero digit
        (1-4) is the highest.
      - We fill that cell with the model's predicted digit.
      - We repeat until the grid is full or no progress can be made.
    """
    model.eval()
    grid = [row[:] for row in start_grid]

    with torch.no_grad():
        for step in range(16):
            # If full, we are done
            if all(grid[r][c] != 0 for r in range(4) for c in range(4)):
                break

            # Convert current grid to tensor
            flat_grid = [grid[r][c] for r in range(4) for c in range(4)]
            inp_tensor = torch.tensor([flat_grid], dtype=torch.long, device=device)

            # Predict logits
            logits = model(inp_tensor).squeeze(0) # [16, 5]
            probs = torch.softmax(logits, dim=-1) # [16, 5]

            best_r, best_c = -1, -1
            best_val = -1
            best_confidence = -1.0

            for idx in range(16):
                r, c = idx // 4, idx % 4
                if grid[r][c] == 0:
                    # Model predictions for non-zero classes (1..4)
                    class_probs = probs[idx, 1:5] # class 0 is empty
                    val_idx = torch.argmax(class_probs).item()
                    val = val_idx + 1 # offset by 1
                    conf = class_probs[val_idx].item()

                    if conf > best_confidence:
                        best_confidence = conf
                        best_val = val
                        best_r, best_c = r, c

            if best_r != -1:
                grid[best_r][best_c] = best_val
            else:
                break

    return grid

def evaluate_on_puzzles(model, solved_test_grids, device='cpu'):
    """
    Generates test puzzles, solves them autoregressively, and reports accuracy.
    """
    solved_count = 0
    total_count = 0

    print("\nStarting evaluation of autoregressive solver on test set puzzles...")
    for idx, solved_grid in enumerate(solved_test_grids):
        # Generate puzzle (first state in the greedy path)
        path = generate_greedy_puzzle_path(solved_grid)
        puzzle = path[0]

        solved_candidate = solve_puzzle_autoregressive(model, puzzle, device)

        # Check if they match the solved_grid perfectly
        is_correct = solved_candidate == solved_grid
        if is_correct:
            solved_count += 1
        total_count += 1

        if idx < 5:
            print(f"\n--- Test Puzzle {idx + 1} ---")
            print("Initial:")
            for r in puzzle:
                print(r)
            print("Model Solved:")
            for r in solved_candidate:
                print(r)
            print("Correct Ground Truth:")
            for r in solved_grid:
                print(r)
            print(f"Outcome: {'SUCCESS' if is_correct else 'FAILURE'}")

    accuracy = (solved_count / total_count) * 100
    print(f"\nSolved {solved_count}/{total_count} puzzles. Accuracy: {accuracy:.2f}%")
    return accuracy

# ---------------------------------------------------------
# Main Execution Script
# ---------------------------------------------------------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate data
    solved_grids = generate_all_solved_grids()
    print(f"Total solved 4x4 grids: {len(solved_grids)}")

    train_dataset, val_dataset, test_grids = build_datasets(solved_grids, train_ratio=0.85, seed=123, augment_factor=5)
    print(f"Train Dataset Size (Transitions): {len(train_dataset)}")
    print(f"Validation Dataset Size (Transitions): {len(val_dataset)}")
    print(f"Test Sets (Completed Grids to Solve): {len(test_grids)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize Model
    model = SudokuTransformer(embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2)

    # Train
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=40, lr=2e-3, device=device)

    # Create directory for charts if it doesn't exist
    os.makedirs('charts', exist_ok=True)

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Transformer 4x4 Sudoku Solver Training Trajectory")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig('charts/sudoku_training_loss.png')
    plt.close()
    print("Training chart saved to 'charts/sudoku_training_loss.png'")

    # Evaluate
    evaluate_on_puzzles(model, test_grids, device)
