# Executive Summary — Tiny Reasoning Models for Scheduling & Constraint Problems

*A plain-English overview for a non-technical audience.*

## The big idea in one line

We showed that a **very small, cheap AI model** — paired with a simple
rule-checker — can solve real scheduling-type problems **100% correctly and
about 5× faster** than the specialist software normally used, cheaply enough to
run on a laptop with no cloud.

## The problem we're addressing

Many business problems are really big puzzles with strict rules: *who works which
shift, which plane goes to which gate, how to split an order across accounts
without breaking any limits.* These are everywhere in IT-services work (BPO,
managed services, operations).

Today they're solved either by **people in spreadsheets** (slow, error-prone) or
by **heavy optimisation software** (correct, but can be slow and expensive at
scale). Meanwhile, large AI models (the ChatGPT family) are **surprisingly bad**
at this kind of strict-rule puzzle and are costly to run.

## What we actually did

There is a new class of "tiny reasoning models" in recent research. The research
papers were full of promising claims but short on **real, reproduced results**,
and none had been tried on an industry problem. So we:

1. **Built one of these tiny models from scratch** (about 220,000 "dials" —
   thousands of times smaller than a typical AI model).
2. **Tested it on a real-style task**: fixing gaps in a staff duty roster
   (someone calls in sick → fill the slot without breaking any rules).
3. **Measured everything honestly** against a trusted rule-checker, on data the
   model had never seen.

## What we found

- **Smart design beats brute size.** The "thinking in loops" design clearly
  outperformed a same-sized ordinary model (big quality jump from the method
  alone, not from being bigger).
- **You can make it better on the fly** by letting it try several answers and
  keeping the one that passes the rules — **no retraining needed.**
- **The winning recipe** is: the tiny model proposes an answer → a rule-checker
  confirms it's valid → and on the few hard cases it can't crack, we fall back to
  the exact (slower) software.
  - Result: **100% correct answers, the expensive software needed only ~1 in 9
    times, and ~5× faster overall.**
- **We also reported where it fails** (it needs clean, well-defined training
  data) — because an honest limitation is more trustworthy than a perfect-sounding
  claim.

## Why this matters — our value add

- **Cost**: tiny model = tiny compute bill. It can run on ordinary hardware.
- **Speed**: fast enough for live, "fix-it-now" situations (e.g. re-planning when
  a flight is cancelled).
- **Trust**: the rule-checker means answers are **correct by construction**, not
  "probably right" — essential in regulated settings.
- **Privacy / edge**: small enough to run **on-site, with no data leaving the
  building** — important for banks and airport operations.
- **Generalises to new cases**: unlike the original research models (which can
  only handle the exact examples they were trained on), ours **learns the rules**
  and handles rosters it has never seen before.
- **Novelty**: we turned unproven research claims into a **working, measured
  demonstration**. A real literature check confirmed that — to our knowledge — no
  one had benchmarked this type of tiny recursive model on an industry scheduling
  task (there is other neural scheduling work, which we cite and build on, but not
  with these models or this verifier-backed design).

## Where it could apply

- **Airlines / travel**: crew scheduling, gate assignment, aircraft-to-route
  planning, and fast disruption recovery when flights cancel.
- **Finance**: splitting trades across accounts within limits, settlement and
  collateral assignment, compliance/reconciliation checks, and operations
  scheduling — all runnable on-premise for data privacy.
- **Not a fit**: anything needing world knowledge or judgement (price
  prediction, credit decisions, customer chat) — those still need large AI.

## Honest caveats

This is an early **proof that the mechanism works**, on small examples, run once
each, on a laptop. It is not yet a production benchmark. The clear next step is a
**scoped pilot** on one real workload (e.g. one scheduling process), comparing
cost, speed and accuracy against the current approach.

## The bottom line

A tiny, cheap, on-device AI model — wrapped with a rule-checker and a safety-net
solver — is a credible new way to handle scheduling and constraint problems:
**provably correct, much faster, and far cheaper** than today's options, with
clear routes into travel and finance.
