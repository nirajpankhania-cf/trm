"""Scoped pilot: rigor (multi-seed), generalization, economic scaling, SLM baseline.

Design constraints: CPU-only, foreground, short runs. Each invocation does ONE
unit of work and appends a JSON row to results/pilot/rows.jsonl, so multi-seed /
multi-size studies are assembled from several short calls and aggregated later.

Commands:
    python -m experiments.pilot yields  --size 6x5 --blanks 6
    python -m experiments.pilot run     --size 6x5 --seed 1 --blanks 6 --epochs 110
    python -m experiments.pilot shift   --size 6x5 --seed 1   # train 6x5, test harder
    python -m experiments.pilot aggregate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trm_industry import checkpoint as CK
from trm_industry import inference as INF
from trm_industry.tasks import rostering as R
from experiments.common import train
from experiments.run import RosterTask

PILOT = Path("results/pilot")
ROWS = PILOT / "rows.jsonl"
PCK = PILOT / "ckpts"


def parse_size(s: str):
    w, d = s.lower().split("x")
    return int(w), int(d)


def make_spec(size: str) -> R.RosterSpec:
    """Sensible coverage/workload that stays feasible as the grid grows."""
    w, d = parse_size(size)
    need_day = max(1, w // 3)
    need_night = 1
    max_shifts = max(2, d - 1)
    return R.RosterSpec(workers=w, days=d, need_day=need_day,
                        need_night=need_night, max_shifts=max_shifts)


def append_row(row: dict):
    PILOT.mkdir(parents=True, exist_ok=True)
    with open(ROWS, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("appended row:", {k: row[k] for k in ("kind", "size", "seed") if k in row}, flush=True)


# --------------------------- commands ---------------------------------
def cmd_yields(size, blanks, n=120):
    spec = make_spec(size)
    rng = np.random.default_rng(0)
    uniq = tot = 0
    t0 = time.perf_counter()
    for _ in range(n):
        full = R.generate_full(spec, rng)
        if full is None:
            continue
        puz = (full + 1).copy()
        idx = rng.permutation(spec.seq_len)[:blanks]
        for k in idx:
            r, c = divmod(int(k), spec.days)
            puz[r, c] = 0
        tot += 1
        if R.count_solutions(puz, spec, 2) == 1:
            uniq += 1
    print(f"size {size} spec(nd={spec.need_day},nn={spec.need_night},ms={spec.max_shifts}) "
          f"blanks {blanks}: unique {uniq}/{tot} ({100*uniq/max(1,tot):.0f}%) "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)


def _train_model(spec, blanks, seed, epochs, nsamp, ntest):
    tr = R.make_dataset(spec, nsamp, blanks, seed=seed, unique=True)
    te_in, te_tg, te_pz, te_sol = R.make_dataset(spec, ntest, blanks, seed=1000 + seed, unique=True)
    te = (te_in, te_pz, te_sol)
    model, _, m = train(RosterTask(), spec, tr, te, recursive=True, hidden=128,
                        n_latent=4, T=2, n_sup=4, epochs=epochs, lr=3e-3,
                        blanks=blanks, seed=seed, eval_every=0, verbose=False)
    return model, m, (te_in, te_pz, te_sol)


def _ckpt_path(size, seed):
    return str(PCK / f"{size}_s{seed}.pt")


def cmd_run(size, seed, blanks, epochs, nsamp=1000, ntest=120, sigma=0.5,
            Ks=(1, 4, 16, 64), hybrid_K=16):
    spec = make_spec(size)
    dev = torch.device("cpu")
    t0 = time.perf_counter()
    model, m, (te_in, te_pz, te_sol) = _train_model(spec, blanks, seed, epochs, nsamp, ntest)
    CK.save(_ckpt_path(size, seed), model, extra={"blanks": blanks, "size": size, "seed": seed})
    sc = INF.scaling_eval(model, te_in, te_pz, te_sol, spec, R.verify, dev,
                          Ks=Ks, sigma=sigma, assemble=R.assemble, grid_shape=spec.shape)
    hyb = INF.hybrid_solve(model, te_in, te_pz, spec, R.verify, R.exact_solve, dev,
                           K=hybrid_K, sigma=sigma, assemble=R.assemble, grid_shape=spec.shape)
    row = {
        "kind": "run", "size": size, "seed": seed, "blanks": blanks, "epochs": epochs,
        "workers": spec.workers, "days": spec.days, "seq_len": spec.seq_len,
        "deterministic_valid": m["valid_solve"], "params": m["params"],
        "train_time_s": m["train_time_s"],
        "scaling": {str(k): sc[k] for k in Ks},
        "hybrid": hyb,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    append_row(row)
    print(f"[{size} seed{seed}] det_valid={m['valid_solve']:.3f} "
          f"verifier@{hybrid_K}={sc[hybrid_K]['verifier']:.3f} "
          f"hybrid_feasible={hyb['hybrid_feasible']:.3f} "
          f"avoided={hyb['solver_calls_avoided']:.3f} speedup={hyb['speedup_vs_solver']:.2f} "
          f"wall={row['wall_s']}s", flush=True)


def cmd_shift(size, seed, blanks, epochs, nsamp=1000, ntest=120, sigma=0.5, hybrid_K=16):
    """Load the trained (size,seed) model; test on harder shifted sets (more blanks)."""
    spec = make_spec(size)
    dev = torch.device("cpu")
    ckp = _ckpt_path(size, seed)
    if Path(ckp).exists():
        model, _ = CK.load(ckp, dev)
        in_dist = None
    else:
        model, m, _ = _train_model(spec, blanks, seed, epochs, nsamp, ntest)
        CK.save(ckp, model, extra={"blanks": blanks, "size": size, "seed": seed})
        in_dist = m["valid_solve"]
    results = {"train_blanks": blanks, "in_dist_valid": in_dist}
    for tb in sorted({blanks, blanks + 2, blanks + 4}):
        te_in, te_tg, te_pz, te_sol = R.make_dataset(spec, ntest, tb, seed=2000 + seed, unique=True)
        det = INF.scaling_eval(model, te_in, te_pz, te_sol, spec, R.verify, dev,
                               Ks=(1, hybrid_K), sigma=sigma, assemble=R.assemble,
                               grid_shape=spec.shape)
        hyb = INF.hybrid_solve(model, te_in, te_pz, spec, R.verify, R.exact_solve, dev,
                               K=hybrid_K, sigma=sigma, assemble=R.assemble, grid_shape=spec.shape)
        results[f"test_blanks_{tb}"] = {
            "verifier@1": det[1]["verifier"], f"verifier@{hybrid_K}": det[hybrid_K]["verifier"],
            "hybrid_feasible": hyb["hybrid_feasible"],
            "solver_calls_avoided": hyb["solver_calls_avoided"],
            "speedup": hyb["speedup_vs_solver"],
        }
    row = {"kind": "shift", "size": size, "seed": seed, **results}
    append_row(row)
    print(f"[shift {size} seed{seed}] {json.dumps(results)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["yields", "run", "shift", "aggregate"])
    ap.add_argument("--size", default="6x5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--blanks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--nsamp", type=int, default=900)
    ap.add_argument("--ntest", type=int, default=120)
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.cmd == "yields":
        cmd_yields(a.size, a.blanks)
    elif a.cmd == "run":
        cmd_run(a.size, a.seed, a.blanks, a.epochs, a.nsamp, a.ntest)
    elif a.cmd == "shift":
        cmd_shift(a.size, a.seed, a.blanks, a.epochs, a.nsamp, a.ntest)
    elif a.cmd == "aggregate":
        from experiments.pilot_aggregate import main as agg_main
        agg_main()
