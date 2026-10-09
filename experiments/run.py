"""Experiment driver. Run as, e.g.:

    python -m experiments.run sudoku
    python -m experiments.run roster
    python -m experiments.run plots

Each subcommand is self-contained and writes JSON + checkpoints under results/.
Kept short enough to complete in a single foreground invocation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from trm_industry import checkpoint as ckpt
from trm_industry import inference as INF
from trm_industry.tasks import rostering as R
from trm_industry.tasks import sudoku as S
from trm_industry.utils import set_seed
from experiments.common import train, RESULTS, CKPTS


# ----------------------------- task adapters ----------------------------
class SudokuTask:
    name = "sudoku"
    verify = staticmethod(S.verify)

    @staticmethod
    def assemble(puz, pred_grid):
        return np.where(puz > 0, puz, pred_grid + 1)

    @staticmethod
    def exact_solve(puz, spec):
        return S.exact_solve(puz, spec)


class RosterTask:
    name = "roster"
    verify = staticmethod(R.verify)
    assemble = staticmethod(R.assemble)
    exact_solve = staticmethod(R.exact_solve)


def _save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("wrote", path, flush=True)


# ------------------------------- Sudoku ---------------------------------
def run_sudoku(n=4, blanks=8, nsamp=2000, ntest=200, epochs=170,
               hidden=128, n_latent=4, T=2, n_sup=4, sigma=0.4,
               Ks=(1, 2, 4, 8, 16, 32, 64)):
    spec = S.SudokuSpec(n=n)
    tr = S.make_dataset(spec, nsamp, blanks, seed=1)
    te_in, te_tg, te_pz, te_sol = S.make_dataset(spec, ntest, blanks, seed=99)
    te = (te_in, te_pz, te_sol)
    task = SudokuTask()

    print("== Sudoku: recursive TRM ==", flush=True)
    rec, _, rec_m = train(task, spec, tr, te, recursive=True, hidden=hidden,
                          n_latent=n_latent, T=T, n_sup=n_sup, epochs=epochs,
                          lr=3e-3, blanks=blanks, eval_every=max(1, epochs // 6))
    print("recursive:", rec_m, flush=True)
    ckpt.save(f"{CKPTS}/sudoku_recursive.pt", rec, extra={"spec_n": n, "blanks": blanks})

    print("== Sudoku: one-pass baseline ==", flush=True)
    onep, _, one_m = train(task, spec, tr, te, recursive=False, hidden=hidden,
                           n_latent=0, T=1, n_sup=1, epochs=epochs,
                           lr=3e-3, blanks=blanks, eval_every=max(1, epochs // 6))
    print("onepass:", one_m, flush=True)

    print("== Sudoku: inference-time scaling (verifier-gated) ==", flush=True)
    dev = torch.device("cpu")
    t0 = time.perf_counter()
    scaling = INF.scaling_eval(rec, te_in, te_pz, te_sol, spec, S.verify, dev,
                               Ks=Ks, sigma=sigma, assemble=SudokuTask.assemble)
    print("scaling done in", round(time.perf_counter() - t0, 1), "s", flush=True)
    for K in Ks:
        print(f"  K={K:3d}  pass {scaling[K]['pass']:.3f}  "
              f"verifier {scaling[K]['verifier']:.3f}  "
              f"bestQ {scaling[K]['bestq']:.3f}  majority {scaling[K]['majority']:.3f}",
              flush=True)

    out = {
        "task": "sudoku", "n": n, "blanks": blanks,
        "recursive": rec_m, "onepass": one_m,
        "scaling": {str(k): v for k, v in scaling.items()},
        "sigma": sigma,
    }
    _save(out, f"{RESULTS}/sudoku.json")
    return out


# ------------------------------ Rostering -------------------------------
ROSTER_BLANKS = 6


def run_roster_train(blanks=ROSTER_BLANKS, nsamp=1400, ntest=200, epochs=175,
                     hidden=128, n_latent=4, T=2, n_sup=4):
    """Train recursive + one-pass roster models, save checkpoints + metrics."""
    spec = R.RosterSpec()
    tr = R.make_dataset(spec, nsamp, blanks, seed=1)
    te_in, te_tg, te_pz, te_sol = R.make_dataset(spec, ntest, blanks, seed=99)
    te = (te_in, te_pz, te_sol)
    task = RosterTask()

    print("== Roster: recursive TRM ==", flush=True)
    rec, _, rec_m = train(task, spec, tr, te, recursive=True, hidden=hidden,
                          n_latent=n_latent, T=T, n_sup=n_sup, epochs=epochs,
                          lr=3e-3, blanks=blanks, eval_every=max(1, epochs // 8))
    print("recursive:", rec_m, flush=True)
    ckpt.save(f"{CKPTS}/roster_recursive.pt", rec, extra={"blanks": blanks})

    print("== Roster: one-pass baseline ==", flush=True)
    onep, _, one_m = train(task, spec, tr, te, recursive=False, hidden=hidden,
                           n_latent=0, T=1, n_sup=1, epochs=epochs,
                           lr=3e-3, blanks=blanks, eval_every=max(1, epochs // 8))
    print("onepass:", one_m, flush=True)

    _save({"task": "roster", "blanks": blanks,
           "spec": {"workers": spec.workers, "days": spec.days,
                    "need_day": spec.need_day, "need_night": spec.need_night,
                    "max_shifts": spec.max_shifts},
           "recursive": rec_m, "onepass": one_m},
          f"{RESULTS}/roster_train.json")


def run_roster_infer(blanks=ROSTER_BLANKS, ntest=200, sigma=0.5,
                     Ks=(1, 2, 4, 8, 16, 32, 64), hybrid_K=16):
    """Load trained roster model; run inference-time scaling + hybrid."""
    dev = torch.device("cpu")
    spec = R.RosterSpec()
    rec, _ = ckpt.load(f"{CKPTS}/roster_recursive.pt", dev)
    te_in, te_tg, te_pz, te_sol = R.make_dataset(spec, ntest, blanks, seed=99)

    print("== Roster: inference-time scaling ==", flush=True)
    scaling = INF.scaling_eval(rec, te_in, te_pz, te_sol, spec, R.verify, dev,
                               Ks=Ks, sigma=sigma, assemble=R.assemble,
                               grid_shape=spec.shape)
    for K in Ks:
        print(f"  K={K:3d}  pass {scaling[K]['pass']:.3f}  "
              f"verifier {scaling[K]['verifier']:.3f}  "
              f"bestQ {scaling[K]['bestq']:.3f}  majority {scaling[K]['majority']:.3f}",
              flush=True)

    print("== Roster: verifier-gated hybrid (TRM + CP-SAT fallback) ==", flush=True)
    hyb = INF.hybrid_solve(rec, te_in, te_pz, spec, R.verify, R.exact_solve, dev,
                           K=hybrid_K, sigma=sigma, assemble=R.assemble,
                           grid_shape=spec.shape)
    print("hybrid:", hyb, flush=True)

    # merge with training metrics if present
    base = {}
    tp = Path(f"{RESULTS}/roster_train.json")
    if tp.exists():
        base = json.loads(tp.read_text())
    base.update({"scaling": {str(k): v for k, v in scaling.items()},
                 "hybrid": hyb, "sigma": sigma})
    _save(base, f"{RESULTS}/roster.json")


def tune_sudoku_scaling(n=4, blanks=8, ntest=200, sigmas=(0.2, 0.3, 0.4, 0.6),
                        Ks=(1, 4, 16, 64)):
    """Reload trained Sudoku model and sweep sigma for inference scaling."""
    dev = torch.device("cpu")
    spec = S.SudokuSpec(n=n)
    rec, _ = ckpt.load(f"{CKPTS}/sudoku_recursive.pt", dev)
    te_in, te_tg, te_pz, te_sol = S.make_dataset(spec, ntest, blanks, seed=99)
    for sg in sigmas:
        sc = INF.scaling_eval(rec, te_in, te_pz, te_sol, spec, S.verify, dev,
                              Ks=Ks, sigma=sg, assemble=SudokuTask.assemble)
        line = "  ".join(f"K{K}:p{sc[K]['pass']:.2f}/v{sc[K]['verifier']:.2f}/q{sc[K]['bestq']:.2f}" for K in Ks)
        print(f"sigma {sg}: {line}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sudoku"
    if cmd == "sudoku":
        run_sudoku()
    elif cmd == "roster_train":
        run_roster_train()
    elif cmd == "roster_infer":
        run_roster_infer()
    elif cmd == "sudoku6":
        run_sudoku(n=6, blanks=14, nsamp=2500, epochs=200, n_latent=6, T=3, n_sup=6)
    elif cmd == "tune_sudoku":
        tune_sudoku_scaling()
    else:
        print("unknown command", cmd)
