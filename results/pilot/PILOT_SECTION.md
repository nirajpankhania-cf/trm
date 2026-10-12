## 13. Scoped Pilot — Rigor, Generalization, Economics, and an SLM Baseline

Section 12 was a single-seed proof of mechanism. This pilot hardens it along the
four axes the earlier caveats demanded. All runs are CPU-only; the tiny recursive
model here is a leaner configuration than Section 12 (trained within a strict
time budget), so absolute accuracy is lower by design — the point is the
*behaviour*, measured with controls.

### 13.1 Statistical rigor (WS1): single-seed numbers are not trustworthy

Across **5 seeds** on the 6x5 roster-repair task, the proposer's
single-shot valid rate is **48% ± 8%** and verifier@16 is
**59% ± 7%** — a spread wide enough that any single-seed
figure (including Section 12's) must be read as indicative, not exact. The
**hybrid feasibility was 100% on every seed**. (Figure `ws1_seeds.png`.)

### 13.2 Generalization under distribution shift (WS2): the system fails safe

Training on 6 blanks and testing on harder instances (more blanks) degrades the
proposer but **never breaks correctness**:

| Test blanks | Proposer verifier@16 (mean) | Hybrid feasibility |
|---|---|---|
| 6 | 60% | 100% |
| 8 | 42% | 100% |
| 10 | 34% | 100% |

The proposer's hit-rate falls as instances get harder than training, yet the
verifier-gated hybrid stays **100% feasible** because it falls back to the exact
solver. This is the key industrial property: **graceful degradation of
efficiency, never of correctness.** (Figure `ws2_shift.png`.)

### 13.3 Economics and scaling (WS3): where the value is — and isn't

| Size | Cells | Proposer valid | CP-SAT ms/inst | TRM ms/inst | Per-call cost ratio | Realized hybrid speedup |
|---|---|---|---|---|---|---|
| 6x5 | 30 | 48% | 0.821 | 0.103 | 8x | 1.92x |
| 8x6 | 48 | 28% | 0.929 | 0.140 | 7x | 1.13x |
| 10x7 | 70 | 2% | 1.230 | 0.156 | 8x | 0.88x |

Two honest conclusions. First, **per attempt the TRM proposer is roughly an order
of magnitude cheaper than a CP-SAT solve at every size.** Second, the **realized**
hybrid speedup *falls* as the grid grows, because under a fixed cheap training
budget the proposer's accuracy collapses faster than CP-SAT slows down — so more
instances hit the fallback. At these sizes CP-SAT is itself sub-millisecond, so
the economic case is not yet made at scale. **The value is real at small/medium
sizes with adequate training; realising it at larger sizes requires
proportionally more training (or a variable-length architecture) and genuinely
hard instances where the exact solver is slow.** (Figure `ws3_size.png`.)

### 13.4 Off-the-shelf SLM baseline (WS4): a small LM cannot do this task

On the same 6x5 test set, a general small language model, prompted few-shot
(no fine-tuning):

| Model | Parse rate | Feasible rate | Time per attempt (ms) |
|---|---|---|---|
| SLM (Qwen2.5-0.5B, few-shot) | 52% | 0% | 3294 |
| TRM proposer (~0.22M, trained) | 100% | 48% | 0.103 |

The 0.5B SLM — over 2,000x larger than the recursive model — produced **0%
feasible rosters** and is **thousands of times slower per attempt**. This is the
honest "off-the-shelf" result the survey said was missing; a *fine-tuned* SLM
would be a fairer and stronger baseline and remains future work. (Figure
`ws4_slm.png`.)

### 13.5 Verdict against the decision rule

The pre-registered rule (Section 9) was: adopt only if the hybrid beats CP-SAT on
cost/latency at matched (100%) feasibility, across seeds and size-shifts.

* **Feasibility:** met — 100% across all seeds and all shift levels.
* **Robustness:** met — correctness is preserved under distribution shift.
* **Cost/latency at scale:** **not yet met** — at these (CP-SAT-easy) sizes and
  under the cheap training budget, realized speedup is modest (and <1 at the
  largest size). 

**Conclusion:** the verifier-gated recursive-proposer + exact-solver pattern is
**sound and safe**, and a tiny recursive model beats an off-the-shelf SLM
decisively, but the **economic case is unproven at scale** and is the explicit
target of the next phase: harder, solver-expensive instances; more training per
size; multiple seeds throughout; and a fine-tuned SLM comparator.
