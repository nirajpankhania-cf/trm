## 12. Original Experiments That Succeeded — A verifier-gated tiny recursive solver, trained and tested

Unlike the failed smoke test in Section 7, the experiments in this section were
run to completion and produced meaningful, reproducible accuracy results on
commodity hardware (Apple-silicon CPU, no GPU). All code is in `trm_industry/`
and `experiments/`; every number below is emitted by `experiments/run.py` and
aggregated by `experiments/report.py`.

The goal is to test, on real runs, the three ideas the literature flags as most
industrially relevant (Sections 5.1, 5.3, 8): (i) recursion with deep
supervision as the performance driver, (ii) verifier-gated inference-time
scaling, and (iii) a verifier-gated recursive proposer behind an exact-solver
fallback — the pattern recommended in Section 8.4.

### 12.1 Setup

* **Model.** A single 2-layer MLP-mixer core (the paper's TRM-MLP variant),
  recursed over two latent states (y, z) with deep recursion (truncated BPTT),
  deep supervision, and an EMA of the weights. ~221K
  parameters. A one-pass model with **identical parameter count** is the
  ablation control.
* **Tasks.** (a) 4x4 Sudoku (8 of 16 cells blanked) as a validated CSP backbone;
  (b) **workforce shift rostering**, 6 workers x 5 days,
  each day requiring exactly 2 day and 1 night
  shift(s), a no-night-then-day rest rule, and at most 4 shifts
  per worker — framed as the "fill-the-gap / shift-swap repair" problem common
  in BPS and managed-services workforce management. Instances are generated and,
  in the fallback path, solved by OR-Tools CP-SAT.
* **Metric.** Valid-solve rate = fraction of test instances for which the
  assembled grid passes the deterministic verifier (coverage/rest/workload for
  rostering; row/column/box for Sudoku).

### 12.2 Finding 1 — Recursion with deep supervision is the driver, at equal parameters

| Task | One-pass (same params) | Recursive TRM | Gain |
|---|---|---|---|
| Sudoku 4x4 (8 blanks) | 81.0% | 99.5% | +18.5 pp |
| Rostering 6x5 (6 blanks) | 58.5% | 81.5% | +23.0 pp |

Both models have the same architecture and parameter count
(~221K); the only difference is whether the core is
recursed with deep supervision. Recursion adds
18 points on Sudoku and
23 points on rostering. This
reproduces, on our own runs, the central claim of Section 4: the gain comes from
iterative refinement, not from parameters or hierarchy.

![Recursion vs one-pass at equal parameters](figs/ablation.png)

### 12.3 Finding 2 — Verifier-gated inference-time scaling, and the Q-head ceiling

Adding Gaussian noise to the latent (sigma=0.4 for Sudoku,
0.5 for rostering) and running K parallel rollouts (PTRM-style), then
selecting with a **deterministic verifier**, lifts the solve rate with no
retraining.

**Sudoku 4x4:**

| K | pass@K | verifier@K | Q-head best@K | majority@K |
|---|---|---|---|---|
| 1 | 98.0 | 98.0 | 98.0 | 98.0 |
| 2 | 100.0 | 100.0 | 98.0 | 98.0 |
| 4 | 100.0 | 100.0 | 98.5 | 100.0 |
| 8 | 100.0 | 100.0 | 98.5 | 99.5 |
| 16 | 100.0 | 100.0 | 98.0 | 99.0 |
| 32 | 100.0 | 100.0 | 98.5 | 99.5 |
| 64 | 100.0 | 100.0 | 98.5 | 99.5 |

**Rostering 6x5:**

| K | pass@K | verifier@K | Q-head best@K | majority@K |
|---|---|---|---|---|
| 1 | 81.0 | 81.0 | 81.0 | 81.0 |
| 2 | 83.0 | 83.0 | 81.0 | 81.0 |
| 4 | 87.5 | 87.5 | 81.5 | 82.5 |
| 8 | 88.0 | 88.0 | 81.0 | 82.0 |
| 16 | 90.0 | 90.0 | 82.0 | 82.0 |
| 32 | 91.0 | 91.0 | 79.5 | 82.0 |
| 64 | 91.5 | 91.5 | 79.5 | 82.0 |

Two things match the literature exactly. First, verifier-gated selection climbs
toward the pass@K oracle as K grows (Sudoku reaches 100%; rostering reaches
91.5%).
Second — and this is the operationally important one — **selection by the learned
Q-head plateaus well below the verifier** (rostering Q-head best@K tops out around
82% while the
verifier reaches
91.5%). This is
the paper's H2 (Section 9): **do not trust the internal ranking head; supply a
deterministic verifier.** Scheduling and constraint problems have one natively.

![Sudoku inference-time scaling](figs/scaling_sudoku.png)

![Rostering inference-time scaling](figs/scaling_roster.png)

### 12.4 Finding 3 — The deployable pattern: verifier-gated proposer + exact-solver fallback

The recommended industrial pattern (Section 8.4) is a tiny recursive proposer,
gated by the verifier, with CP-SAT as the fallback. On 200 held-out rostering
instances with K=16 rollouts:

| Configuration | Feasibility | Solver calls | Total time (200 inst.) |
|---|---|---|---|
| CP-SAT only (exact) | 100% | 200 / 200 | 131 ms |
| TRM proposer only | 89.0% | 0 | 10 ms |
| **Verifier-gated hybrid** | **100.0%** | 22 / 200 | 26 ms |

The hybrid is **sound by construction** (every returned roster passes the
verifier or is produced by the exact solver), achieves **100%
feasibility**, avoids the expensive solver on **89%**
of instances, and is **5.0x faster** end-to-end than
calling CP-SAT on every instance.

![Verifier-gated hybrid: feasibility and latency](figs/hybrid_roster.png)

### 12.5 Finding 4 — Honest limitation: supervised TRM needs label-consistent data

Trained directly on the *general* rostering distribution (interchangeable
workers, multiple feasible completions per puzzle, 8 blanks), the supervised TRM
collapsed to ~2% valid-solve: per-cell supervision cannot fit a one-to-many
mapping. Learnable results required filtering to **unique-completion** instances
(verified by CP-SAT solution counting). This is a concrete, measured instance of
the transductivity caveat in Sections 4 and 8: these are narrow, data-sensitive
solvers, and the verifier/solver — not the network — is what guarantees
correctness in deployment.

### 12.6 What this means for Coforge

For fixed-size, constraint-structured, solver-labelable problems — shift
rostering, timetabling, slotting, configuration validation — a ~221K-parameter
recursive proposer behind a deterministic verifier and a CP-SAT fallback is a
credible, **CPU-deployable, edge-friendly** pattern: it preserves 100% feasibility,
removes the majority of expensive solver calls, and runs several times faster
end-to-end. The evidence here is on small instances and must be scaled and
piloted per the protocol in Section 9 before any production claim; but it is a
working, measured demonstration rather than an assertion.

### 12.7 Industrial applicability beyond rostering: finance and travel

The result is not about rostering specifically; it is about a **reusable
pattern**. Any problem that is (i) a fixed-size set of decisions, (ii) governed
by hard rules that can be cheaply checked, (iii) available in many similar
instances, and (iv) under latency, cost, privacy or edge pressure, fits the same
tiny-proposer + verifier + exact-solver-fallback architecture. Two high-value
sectors illustrate the point.

**Travel / airlines** are the textbook home of constraint scheduling, and the
rostering demo is a miniature of several of their core problems:

| Use case | Why it fits | Relationship to our demo |
|---|---|---|
| Crew rostering & pairing | Duty-time limits, rest rules, qualifications; cheap to verify | The same problem, scaled up |
| Gate / stand assignment | Gates x time slots, no clashes, towing times | Direct grid-CSP |
| Tail assignment | Aircraft-to-route under maintenance windows | Fixed-size assignment |
| Disruption recovery / rebooking | Must re-solve in seconds when a flight cancels | Where the latency win is the product |

**Finance** fits best on the rule-checking and allocation problems (not
prediction or world-knowledge tasks):

| Use case | Why it fits | Relationship to our demo |
|---|---|---|
| Trade / order allocation | Split block orders under position limits, lot sizes, mandates | Fixed-size assignment under hard rules |
| Settlement netting & collateral assignment | Eligibility rules; cheap to verify | Constraint satisfaction |
| Pre-trade compliance / reconciliation validation | Check baskets/transactions against regulatory + mandate rules | "Configuration validation" fit |
| Desk / operations shift scheduling | Identical structure to the demo | The same problem |

A property that matters to both sectors: a ~220K-parameter model runs on a
laptop or edge device with **no cloud dependency and no data leaving the
premises** — directly relevant to bank data-residency requirements and to
offline, low-latency airport-operations use.

**Honest scope.** We have demonstrated the *mechanism* on small rostering and
Sudoku instances. Finance and airline problems are structurally identical but
larger and messier; the credible next step is a scoped pilot on one such
workload, measuring feasibility and cost against the incumbent solver (Section 9).
Problems needing world knowledge or continuous optimisation — price prediction,
credit decisioning, demand forecasting, natural-language customer handling — are
**not** a fit for this pattern and should use LLM/ML or classical optimisation.
