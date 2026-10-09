# Tiny Recursive Reasoning Models for Industrial Constraint Satisfaction

A compact, **honest, CPU-runnable** reproduction and extension of TRM-style
recursive reasoning, built to turn the companion survey paper's *proposed*
evaluation protocol (its Section 9) into **actual, completed experiments** with
real accuracy numbers — and to test whether the approach carries industrial
value (Coforge: BPS / managed-services / workforce management).

Everything here runs on a laptop CPU. No GPU required.

## Headline results (all measured, reproducible)

| Result | Number |
|---|---|
| Recursive TRM vs one-pass, **same params**, Sudoku 4x4 | 99.5% vs 81.0% valid |
| Recursive TRM vs one-pass, **same params**, Rostering 6x5 | 81.5% vs 58.5% valid |
| Verifier-gated inference scaling, Sudoku | 98% (K=1) -> 100% (K>=2) |
| Verifier-gated inference scaling, Rostering | 81% (K=1) -> 91.5% (K=64) |
| Q-head selection ceiling vs verifier (rostering) | ~82% vs 91.5% |
| **Verifier-gated hybrid (TRM + CP-SAT fallback)** | **100% feasible, 89% of solver calls avoided, 5.0x faster** |

See `results/RESULTS_SECTION.md` for the drop-in paper section and
`results/figs/` for the figures.

## What this demonstrates

1. **Recursion + deep supervision is the driver** (not parameters or hierarchy):
   the recursive model beats a parameter-identical one-pass model by 18-23 points.
2. **Verifier-gated inference-time scaling** (PTRM-style noise + K rollouts)
   lifts accuracy with no retraining — but **only** when selection uses a
   deterministic verifier; the learned Q-head plateaus well below it.
3. **The deployable pattern** for industry is a tiny recursive proposer gated by
   a verifier, with an exact solver (CP-SAT) as fallback: sound by construction,
   100% feasible, far fewer expensive solver calls, several times faster.
4. **Honest limitation:** supervised TRM needs label-consistent (near-unique)
   training targets; on the general multi-solution rostering distribution it
   collapses to ~2%. The verifier/solver, not the net, guarantees correctness.

## Layout

```
trm_industry/
  model.py         TRM-style recursive model (MLP-mixer core) + one-pass baseline
  train.py         deep-supervision training loop, EMA, evaluation
  inference.py     PTRM-style scaling + verifier-gated hybrid
  checkpoint.py    save/load
  tasks/
    sudoku.py      generator, verifier, exact solver
    rostering.py   CP-SAT generator/solver, verifier, uniqueness filter
experiments/
  common.py        shared training routine
  run.py           experiment driver (CLI)
  report.py        aggregate -> plots + RESULTS_SECTION.md
results/           JSON metrics, checkpoints, figures, paper section
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m experiments.run sudoku          # ~8 min: train + ablation + scaling
python -m experiments.run roster_train    # ~12 min: train recursive + one-pass
python -m experiments.run roster_infer    # ~1 min: scaling + hybrid
python -m experiments.report              # plots + results/RESULTS_SECTION.md
```

Each training run is self-contained and saves a checkpoint, so inference
(`roster_infer`, `tune_sudoku`) can be re-run without retraining.

## Caveats

Small instances (4x4 Sudoku; 6x5 rostering), single seed per configuration,
CPU-only. These are proof-of-mechanism results that realise the survey's
protocol end-to-end, not a production benchmark. Scale and multi-seed per the
protocol in the paper's Section 9 before any deployment claim.
