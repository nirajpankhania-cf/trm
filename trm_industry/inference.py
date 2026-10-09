"""Inference-time scaling (PTRM-style) and verifier-gated hybrid.

These realise the two ideas the paper flags as most industrially relevant:

1. Test-time search over a tiny model's latent trajectories (PTRM: add
   Gaussian noise to the latent, run K rollouts, select among them).
2. A verifier-gated proposer with an exact-solver fallback, which guarantees
   feasibility while only paying for the expensive solver when the cheap
   recursive proposer fails.

Selection modes let us test the paper's finding (H2) that a *deterministic
verifier* beats the learned Q-head for selection.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
import torch


@torch.no_grad()
def _rollout_candidates(model, x_batch, device, K, sigma, n_sup=None):
    """Return preds [B,K,L] (0-indexed classes) and qprob [B,K]."""
    B, L = x_batch.shape
    xr = x_batch.unsqueeze(1).expand(B, K, L).reshape(B * K, L).to(device)
    logits, q = model.solve(xr, noise_sigma=sigma, n_sup=n_sup)
    preds = logits.argmax(dim=-1).reshape(B, K, L).cpu().numpy()
    qp = q.reshape(B, K).cpu().numpy()
    return preds, qp


def _sudoku_assemble(puz, pred_classes_grid):
    return np.where(puz > 0, puz, pred_classes_grid + 1)


def scaling_eval(model, x, puzzles, sols, spec, verify_fn, device,
                 Ks=(1, 2, 4, 8, 16, 32), sigma=0.3, n_sup=None,
                 assemble=_sudoku_assemble, grid_shape=None):
    """Evaluate selection strategies as a function of K rollouts.

    Returns dict: {K: {pass, verifier, bestq, majority}} success rates, where
    success means the *selected* grid is accepted by the deterministic verifier.
    """
    shape = grid_shape if grid_shape is not None else (spec.n, spec.n)
    Kmax = max(Ks)
    preds, qp = _rollout_candidates(model, x, device, Kmax, sigma, n_sup)
    B = preds.shape[0]

    # Precompute validity of every candidate.
    valid = np.zeros((B, Kmax), dtype=bool)
    grids = np.empty((B, Kmax), dtype=object)
    for i in range(B):
        puz = puzzles[i]
        for k in range(Kmax):
            g = assemble(puz, preds[i, k].reshape(shape))
            grids[i, k] = g
            valid[i, k] = verify_fn(g, puz, spec)

    out = {}
    for K in Ks:
        pass_hits = verifier_hits = bestq_hits = maj_hits = 0
        for i in range(B):
            vk = valid[i, :K]
            qk = qp[i, :K]
            # pass@K / verifier@K: any valid candidate (verifier picks it)
            any_valid = bool(vk.any())
            pass_hits += any_valid
            verifier_hits += any_valid
            # bestQ@K: take highest-Q candidate, success iff it is valid
            bq = int(np.argmax(qk))
            bestq_hits += bool(vk[bq])
            # majority@K: most common full-grid, success iff valid
            keys = [grids[i, k].tobytes() for k in range(K)]
            maj_key = Counter(keys).most_common(1)[0][0]
            maj_idx = keys.index(maj_key)
            maj_hits += bool(vk[maj_idx])
        out[K] = {
            "pass": pass_hits / B,
            "verifier": verifier_hits / B,
            "bestq": bestq_hits / B,
            "majority": maj_hits / B,
        }
    return out


@torch.no_grad()
def hybrid_solve(model, x, puzzles, spec, verify_fn, exact_solver, device,
                 K=8, sigma=0.3, n_sup=None,
                 assemble=_sudoku_assemble, grid_shape=None):
    """Verifier-gated proposer + exact-solver fallback.

    For each instance: run K recursive rollouts; if the verifier accepts any,
    return it (cheap). Otherwise call the exact solver (expensive fallback).

    Returns metrics incl. feasibility (should be ~100%), fraction solved by the
    proposer alone (= expensive solver calls avoided), and timing.
    """
    shape = grid_shape if grid_shape is not None else (spec.n, spec.n)
    preds, _ = _rollout_candidates(model, x, device, K, sigma, n_sup)
    B = preds.shape[0]

    solved_by_trm = 0
    feasible = 0
    solver_calls = 0
    t_prop = 0.0
    t_solver = 0.0
    for i in range(B):
        puz = puzzles[i]
        found = False
        t0 = time.perf_counter()
        for k in range(K):
            g = assemble(puz, preds[i, k].reshape(shape))
            if verify_fn(g, puz, spec):
                found = True
                break
        t_prop += time.perf_counter() - t0
        if found:
            solved_by_trm += 1
            feasible += 1
        else:
            solver_calls += 1
            t0 = time.perf_counter()
            sol = exact_solver(puz, spec)
            t_solver += time.perf_counter() - t0
            if sol is not None and verify_fn(sol, puz, spec):
                feasible += 1

    # Pure-solver baseline timing (solve every instance from scratch).
    t0 = time.perf_counter()
    for i in range(B):
        exact_solver(puzzles[i], spec)
    t_solver_only = time.perf_counter() - t0

    return {
        "n": B,
        "K": K,
        "sigma": sigma,
        "trm_only_feasible": solved_by_trm / B,
        "hybrid_feasible": feasible / B,
        "solver_call_rate": solver_calls / B,
        "solver_calls_avoided": 1 - solver_calls / B,
        "time_proposer_s": t_prop,
        "time_fallback_solver_s": t_solver,
        "time_hybrid_total_s": t_prop + t_solver,
        "time_solver_only_s": t_solver_only,
        "speedup_vs_solver": t_solver_only / max(1e-9, t_prop + t_solver),
    }
