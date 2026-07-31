import unittest
import torch
from sudoku.solver import (
    generate_all_solved_grids,
    is_valid_sudoku,
    get_candidates,
    find_naked_singles,
    generate_greedy_puzzle_path,
    SudokuTransformer,
    solve_puzzle_autoregressive
)

class TestSudoku(unittest.TestCase):
    def test_solved_grids_generation(self):
        solved = generate_all_solved_grids()
        self.assertEqual(len(solved), 288)
        for grid in solved:
            self.assertTrue(is_valid_sudoku(grid))

    def test_candidates_and_naked_singles(self):
        # A simple valid 4x4 grid with some missing cells
        grid = [
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 0]  # last element missing, must be 1
        ]
        candidates = get_candidates(grid, 3, 3)
        self.assertEqual(candidates, {1})

        singles = find_naked_singles(grid)
        self.assertEqual(singles, [(3, 3, 1)])

    def test_generate_greedy_puzzle_path(self):
        solved = generate_all_solved_grids()
        g = solved[0]
        path = generate_greedy_puzzle_path(g)
        self.assertTrue(len(path) > 1)
        # Check that path elements are valid transitions
        for i in range(len(path) - 1):
            curr = path[i]
            nxt = path[i+1]
            # Next should have exactly one more filled cell than current, which is a naked single in curr
            diff_count = 0
            single_pos = None
            for r in range(4):
                for c in range(4):
                    if curr[r][c] != nxt[r][c]:
                        self.assertEqual(curr[r][c], 0)
                        diff_count += 1
                        single_pos = (r, c, nxt[r][c])
            self.assertEqual(diff_count, 1)
            # The difference should indeed be a naked single in curr
            singles = find_naked_singles(curr)
            self.assertIn(single_pos, singles)

    def test_transformer_forward(self):
        model = SudokuTransformer(embed_dim=16, num_heads=1, hidden_dim=32, num_layers=1)
        # Input shape [Batch=2, Sequence=16]
        x = torch.zeros((2, 16), dtype=torch.long)
        out = model(x)
        self.assertEqual(out.shape, (2, 16, 5))

if __name__ == '__main__':
    unittest.main()
