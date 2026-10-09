"""Workforce shift rostering as a fixed-size constraint-satisfaction task.

Industrial framing (Coforge BPS / managed-services / workforce management):
given a partially specified duty roster (some cells fixed by availability,
leave, or a dropped shift that must be re-covered), produce a *feasible*
completion. This is the day-to-day "fill the gap / shift-swap repair" problem.

Grid: W workers x D days. Each cell is a shift class:
    0 = OFF, 1 = DAY, 2 = NIGHT.
Token encoding for the model input: 0 = blank (to fill), else class+1
(1=OFF, 2=DAY, 3=NIGHT). So vocab_size = 4, num_classes = 3.

Hard constraints (global, fixed for all instances so the net can learn them):
    * coverage: each day has exactly ``need_day`` DAY and ``need_night`` NIGHT;
    * rest rule: no worker does NIGHT then DAY on consecutive days;
    * workload: each worker does at most ``max_shifts`` working shifts.

A deterministic verifier checks all of the above plus consistency with givens.
CP-SAT (OR-Tools) both generates feasible rosters and acts as the exact
fallback solver in the hybrid pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ortools.sat.python import cp_model

OFF, DAY, NIGHT = 0, 1, 2


@dataclass
class RosterSpec:
    workers: int = 6
    days: int = 5
    need_day: int = 2
    need_night: int = 1
    max_shifts: int = 4

    @property
    def vocab_size(self) -> int:
        return 4  # blank, OFF, DAY, NIGHT

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def seq_len(self) -> int:
        return self.workers * self.days

    @property
    def shape(self):
        return (self.workers, self.days)


def _build_model(spec: RosterSpec, givens: np.ndarray | None):
    """givens: W x D array of class values or -1 for free. Returns (model, day, night)."""
    m = cp_model.CpModel()
    W, D = spec.workers, spec.days
    day = {}
    night = {}
    off = {}
    for w in range(W):
        for d in range(D):
            day[w, d] = m.NewBoolVar(f"day_{w}_{d}")
            night[w, d] = m.NewBoolVar(f"night_{w}_{d}")
            off[w, d] = m.NewBoolVar(f"off_{w}_{d}")
            m.Add(day[w, d] + night[w, d] + off[w, d] == 1)
    # coverage
    for d in range(D):
        m.Add(sum(day[w, d] for w in range(W)) == spec.need_day)
        m.Add(sum(night[w, d] for w in range(W)) == spec.need_night)
    # rest rule: night then day forbidden
    for w in range(W):
        for d in range(D - 1):
            m.Add(night[w, d] + day[w, d + 1] <= 1)
    # workload
    for w in range(W):
        m.Add(sum(day[w, d] + night[w, d] for d in range(D)) <= spec.max_shifts)
    # givens
    if givens is not None:
        for w in range(W):
            for d in range(D):
                g = givens[w, d]
                if g == DAY:
                    m.Add(day[w, d] == 1)
                elif g == NIGHT:
                    m.Add(night[w, d] == 1)
                elif g == OFF:
                    m.Add(off[w, d] == 1)
    return m, day, night, off


def _extract(solver, spec, day, night) -> np.ndarray:
    W, D = spec.workers, spec.days
    grid = np.zeros((W, D), dtype=np.int64)
    for w in range(W):
        for d in range(D):
            if solver.Value(day[w, d]):
                grid[w, d] = DAY
            elif solver.Value(night[w, d]):
                grid[w, d] = NIGHT
            else:
                grid[w, d] = OFF
    return grid


def generate_full(spec: RosterSpec, rng: np.random.Generator) -> np.ndarray | None:
    """Generate one feasible full roster (randomised objective for diversity)."""
    m, day, night, off = _build_model(spec, None)
    # random linear objective to diversify solutions
    terms = []
    for w in range(spec.workers):
        for d in range(spec.days):
            terms.append(int(rng.integers(0, 100)) * day[w, d])
            terms.append(int(rng.integers(0, 100)) * night[w, d])
    m.Maximize(sum(terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    solver.parameters.random_seed = int(rng.integers(0, 1_000_000))
    solver.parameters.num_search_workers = 1
    status = solver.Solve(m)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _extract(solver, spec, day, night)
    return None


def exact_solve(puzzle_tokens: np.ndarray, spec: RosterSpec) -> np.ndarray | None:
    """Complete a blanked roster. puzzle_tokens: W x D tokens (0=blank,1..3).

    Returns class grid (0..2) or None if infeasible.
    """
    givens = np.where(puzzle_tokens > 0, puzzle_tokens - 1, -1)
    m, day, night, off = _build_model(spec, givens)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    solver.parameters.num_search_workers = 1
    status = solver.Solve(m)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _extract(solver, spec, day, night)
    return None


def count_solutions(puzzle_tokens: np.ndarray, spec: RosterSpec, limit: int = 2) -> int:
    """Count feasible completions up to ``limit`` (for uniqueness filtering)."""
    givens = np.where(puzzle_tokens > 0, puzzle_tokens - 1, -1)
    m, day, night, off = _build_model(spec, givens)

    class _Counter(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.count = 0

        def on_solution_callback(self):
            self.count += 1
            if self.count >= limit:
                self.StopSearch()

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 2.0
    cb = _Counter()
    solver.Solve(m, cb)
    return cb.count


def verify(answer_classes: np.ndarray, puzzle_tokens: np.ndarray, spec: RosterSpec) -> bool:
    """True iff answer satisfies all hard constraints and matches givens."""
    W, D = spec.workers, spec.days
    a = answer_classes
    if a.shape != (W, D) or a.min() < 0 or a.max() > 2:
        return False
    givens = np.where(puzzle_tokens > 0, puzzle_tokens - 1, -1)
    mask = givens >= 0
    if not (a[mask] == givens[mask]).all():
        return False
    for d in range(D):
        if int((a[:, d] == DAY).sum()) != spec.need_day:
            return False
        if int((a[:, d] == NIGHT).sum()) != spec.need_night:
            return False
    for w in range(W):
        for d in range(D - 1):
            if a[w, d] == NIGHT and a[w, d + 1] == DAY:
                return False
        if int(((a[w] == DAY) | (a[w] == NIGHT)).sum()) > spec.max_shifts:
            return False
    return True


def assemble(puzzle_tokens: np.ndarray, pred_classes_grid: np.ndarray) -> np.ndarray:
    """Combine givens with predicted classes into a full class grid (0..2)."""
    givens = np.where(puzzle_tokens > 0, puzzle_tokens - 1, -1)
    return np.where(givens >= 0, givens, pred_classes_grid)


def make_dataset(spec: RosterSpec, n_samples: int, num_blanks: int, seed: int,
                 unique: bool = True):
    """Return (inputs[B,L] tokens, targets[B,L] classes, puzzles np, sols np).

    With ``unique=True`` only puzzles whose completion is unique are kept, so
    the supervised label is consistent (otherwise interchangeable workers make
    the task unlearnable by per-cell supervision).
    """
    rng = np.random.default_rng(seed)
    W, D = spec.workers, spec.days
    L = spec.seq_len
    inputs = np.zeros((n_samples, L), dtype=np.int64)
    targets = np.zeros((n_samples, L), dtype=np.int64)
    puzzles, sols = [], []
    made = 0
    attempts = 0
    while made < n_samples and attempts < n_samples * 60:
        attempts += 1
        full = generate_full(spec, rng)
        if full is None:
            continue
        tokens = full + 1  # class->token
        puz = tokens.copy()
        idx = rng.permutation(W * D)[:num_blanks]
        for k in idx:
            r, c = divmod(int(k), D)
            puz[r, c] = 0
        if unique and count_solutions(puz, spec, limit=2) != 1:
            continue
        inputs[made] = puz.flatten()
        targets[made] = full.flatten()
        puzzles.append(puz)
        sols.append(full)
        made += 1
    if made < n_samples:
        inputs = inputs[:made]
        targets = targets[:made]
    return (
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        np.stack(puzzles),
        np.stack(sols),
    )
