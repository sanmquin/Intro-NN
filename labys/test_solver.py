import unittest
import torch
from labys.solver import (
    generate_labyrinth,
    solve_bfs,
    generate_transitions_from_shortest_path,
    LabyrinthDataset,
    LabyrinthTransformer,
    solve_labyrinth_autoregressive
)

class TestLabyrinthSolver(unittest.TestCase):
    def setUp(self):
        # Generate a standard labyrinth for testing
        self.start = (0, 0)
        self.end = (9, 9)
        self.grid = generate_labyrinth(width=10, height=10, start=self.start, end=self.end, num_extra_paths=10)

    def test_labyrinth_boundaries_and_values(self):
        # Grid must be 10x10
        self.assertEqual(len(self.grid), 10)
        self.assertEqual(len(self.grid[0]), 10)

        # Grid elements must be within 0-9
        for r in range(10):
            for c in range(10):
                val = self.grid[r][c]
                self.assertTrue(0 <= val <= 9, f"Grid value {val} at ({r}, {c}) is out of bounds")

        # Start and end positions must be labeled 1 and 2 respectively
        self.assertEqual(self.grid[self.start[0]][self.start[1]], 1)
        self.assertEqual(self.grid[self.end[0]][self.end[1]], 2)

    def test_bfs_shortest_path(self):
        # BFS must find a valid path from start to end
        path = solve_bfs(self.grid, start=self.start, end=self.end)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.end)

        # Verify connectivity of steps in path
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            dist = abs(r1 - r2) + abs(c1 - c2)
            self.assertEqual(dist, 1)
            self.assertIn(self.grid[r2][c2], (0, 1, 2))

    def test_transition_generation(self):
        path = solve_bfs(self.grid, start=self.start, end=self.end)
        transitions = generate_transitions_from_shortest_path(self.grid, path)

        # Number of transitions must be length of path - 1
        self.assertEqual(len(transitions), len(path) - 1)

        # Check transition content
        flat_grid, curr_idx, next_idx = transitions[0]
        self.assertEqual(len(flat_grid), 100)
        self.assertEqual(curr_idx, self.start[0] * 10 + self.start[1])

    def test_transformer_forward_pass(self):
        model = LabyrinthTransformer(embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2)
        grid_tensor = torch.zeros((4, 100), dtype=torch.long)
        curr_pos_tensor = torch.zeros((4,), dtype=torch.long)

        logits = model(grid_tensor, curr_pos_tensor)
        self.assertEqual(logits.shape, (4, 100))

if __name__ == '__main__':
    unittest.main()
