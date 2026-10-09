"""Sudoku as a validated constraint-satisfaction backbone.

Supports N in {4, 6, 9} (box dims inferred). Provides:
* a randomized full-solution generator (backtracking);
* puzzle creation by blanking cells;
* a deterministic verifier (the native "cheap exact verifier" the paper argues
  industrial problems must have);
* an exact backtracking solver used as the hybrid fallback.

Encoding: input token 0 = blank, 1..N = given digit (vocab_size = N+1).
Target class index = digit-1 (num_classes = N). seq_len = N*N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def box_dims(n: int) -> tuple[int, int]:
    return {4: (2, 2), 6: (2, 3), 9: (3, 3)}[n]


@dataclass
class SudokuSpec:
    n: int = 4

    @property
    def box(self) -> tuple[int, int]:
        return box_dims(self.n)

    @property
    def vocab_size(self) -> int:
        return self.n + 1

    @property
    def num_classes(self) -> int:
        return self.n

    @property
    def seq_len(self) -> int:
        return self.n * self.n


def _full_solution(spec: SudokuSpec, rng: np.random.Generator) -> np.ndarray:
    n = spec.n
    br, bc = spec.box
    grid = np.zeros((n, n), dtype=np.int64)

    def ok(r, c, v):
        if (grid[r, :] == v).any() or (grid[:, c] == v).any():
            return False
        r0, c0 = (r // br) * br, (c // bc) * bc
        return not (grid[r0:r0 + br, c0:c0 + bc] == v).any()

    def fill(pos):
        if pos == n * n:
            return True
        r, c = divmod(pos, n)
        vals = rng.permutation(np.arange(1, n + 1))
        for v in vals:
            if ok(r, c, int(v)):
                grid[r, c] = int(v)
                if fill(pos + 1):
                    return True
                grid[r, c] = 0
        return False

    fill(0)
    return grid


def make_puzzle(solution: np.ndarray, num_blanks: int, rng: np.random.Generator) -> np.ndarray:
    n = solution.shape[0]
    puzzle = solution.copy()
    idx = rng.permutation(n * n)[:num_blanks]
    for k in idx:
        r, c = divmod(int(k), n)
        puzzle[r, c] = 0
    return puzzle


def verify(grid: np.ndarray, givens: np.ndarray, spec: SudokuSpec) -> bool:
    """True iff grid is a fully-filled valid Sudoku consistent with givens."""
    n = spec.n
    br, bc = spec.box
    if grid.shape != (n, n):
        return False
    if (grid < 1).any() or (grid > n).any():
        return False
    mask = givens > 0
    if not (grid[mask] == givens[mask]).all():
        return False
    target = set(range(1, n + 1))
    for i in range(n):
        if set(grid[i, :].tolist()) != target:
            return False
        if set(grid[:, i].tolist()) != target:
            return False
    for r0 in range(0, n, br):
        for c0 in range(0, n, bc):
            if set(grid[r0:r0 + br, c0:c0 + bc].flatten().tolist()) != target:
                return False
    return True


def exact_solve(puzzle: np.ndarray, spec: SudokuSpec) -> np.ndarray | None:
    """Backtracking exact solver (the deterministic fallback)."""
    n = spec.n
    br, bc = spec.box
    grid = puzzle.copy()

    def ok(r, c, v):
        if (grid[r, :] == v).any() or (grid[:, c] == v).any():
            return False
        r0, c0 = (r // br) * br, (c // bc) * bc
        return not (grid[r0:r0 + br, c0:c0 + bc] == v).any()

    def find_empty():
        # most-constrained-variable heuristic for speed
        best, best_r, best_c = n + 1, -1, -1
        for r in range(n):
            for c in range(n):
                if grid[r, c] == 0:
                    cnt = sum(1 for v in range(1, n + 1) if ok(r, c, v))
                    if cnt < best:
                        best, best_r, best_c = cnt, r, c
                        if cnt <= 1:
                            return best_r, best_c
        return best_r, best_c

    def solve():
        r, c = find_empty()
        if r == -1:
            return True
        for v in range(1, n + 1):
            if ok(r, c, v):
                grid[r, c] = v
                if solve():
                    return True
                grid[r, c] = 0
        return False

    return grid if solve() else None


def make_dataset(spec: SudokuSpec, n_samples: int, num_blanks: int, seed: int):
    """Return (inputs LongTensor[B,L], targets LongTensor[B,L], puzzles np, sols np)."""
    rng = np.random.default_rng(seed)
    L = spec.seq_len
    inputs = np.zeros((n_samples, L), dtype=np.int64)
    targets = np.zeros((n_samples, L), dtype=np.int64)
    puzzles, sols = [], []
    for i in range(n_samples):
        sol = _full_solution(spec, rng)
        puz = make_puzzle(sol, num_blanks, rng)
        inputs[i] = puz.flatten()
        targets[i] = sol.flatten() - 1  # class index 0..n-1
        puzzles.append(puz)
        sols.append(sol)
    return (
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        np.stack(puzzles),
        np.stack(sols),
    )
