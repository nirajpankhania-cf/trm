"""SLM baseline arm (WS4): a small instruction-tuned language model on the
same rostering task, rendered as text. Few-shot, no fine-tuning (CPU-only).

This fills the gap the survey flagged (no SLM benchmark on these constraint
tasks). We measure: parse rate, feasibility (verifier), and latency, on the
same test distribution used by the TRM pilot, for an honest head-to-head.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from trm_industry.tasks import rostering as R
from experiments.pilot import make_spec, PILOT, append_row

LET = {R.OFF: "O", R.DAY: "D", R.NIGHT: "N"}
TOK = {0: "?", 1: "O", 2: "D", 3: "N"}


def render_puzzle(puz_tokens, spec):
    rows = []
    for w in range(spec.workers):
        rows.append(" ".join(TOK[int(puz_tokens[w, d])] for d in range(spec.days)))
    return "\n".join(f"W{w}: {rows[w]}" for w in range(spec.workers))


def render_solution(classes, spec):
    rows = []
    for w in range(spec.workers):
        rows.append(" ".join(LET[int(classes[w, d])] for d in range(spec.days)))
    return "\n".join(f"W{w}: {rows[w]}" for w in range(spec.workers))


def parse_grid(text, spec):
    cells = re.findall(r"[ODN?]", text.upper())
    cells = [c for c in cells if c in "ODN"]  # drop any stray '?'
    need = spec.workers * spec.days
    if len(cells) < need:
        return None
    cells = cells[:need]
    inv = {"O": R.OFF, "D": R.DAY, "N": R.NIGHT}
    arr = np.array([inv[c] for c in cells], dtype=np.int64).reshape(spec.workers, spec.days)
    return arr


def build_prompt(spec, example_puz, example_sol, query_puz):
    rules = (
        f"You schedule {spec.workers} workers (W0..W{spec.workers-1}) over "
        f"{spec.days} days. Each cell is a shift: O=off, D=day, N=night.\n"
        f"Hard rules:\n"
        f"- Every day has EXACTLY {spec.need_day} D and {spec.need_night} N across workers.\n"
        f"- A worker must NOT do N then D on consecutive days.\n"
        f"- Each worker does at most {spec.max_shifts} working (D or N) shifts.\n"
        f"Fill every '?' so all rules hold. Keep given cells unchanged.\n"
        f"Output ONLY the completed grid, one line per worker as 'W#: c c c ...'.\n"
    )
    ex = (f"Example puzzle:\n{render_puzzle(example_puz, spec)}\n"
          f"Example answer:\n{render_solution(example_sol, spec)}\n\n")
    q = f"Now solve this puzzle:\n{render_puzzle(query_puz, spec)}\nAnswer:\n"
    return rules + ex + q


def run_slm(size="6x5", n=50, seed=1, blanks=6, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    spec = make_spec(size)
    # same test distribution as the pilot run for this (size, seed)
    te_in, te_tg, te_pz, te_sol = R.make_dataset(spec, n, blanks, seed=1000 + seed, unique=True)
    # one solved example for the few-shot anchor
    ex_in, ex_tg, ex_pz, ex_sol = R.make_dataset(spec, 1, blanks, seed=7, unique=True)

    print("loading", model_id, flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    feasible = parsed = 0
    t_gen = 0.0
    N = te_pz.shape[0]
    for i in range(N):
        prompt = build_prompt(spec, ex_pz[0], ex_sol[0], te_pz[i])
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        attn = torch.ones_like(ids)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=attn, max_new_tokens=spec.seq_len * 3 + 20,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        t_gen += time.perf_counter() - t0
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        grid = parse_grid(txt, spec)
        if grid is not None:
            parsed += 1
            if R.verify(grid, te_pz[i], spec):
                feasible += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N} parsed={parsed} feasible={feasible} "
                  f"avg_gen={t_gen/(i+1):.2f}s", flush=True)

    row = {
        "kind": "slm", "size": size, "seed": seed, "model": model_id, "n": N,
        "parse_rate": parsed / N, "feasible_rate": feasible / N,
        "avg_gen_s": t_gen / N, "total_gen_s": round(t_gen, 1),
    }
    append_row(row)
    print(f"[SLM {model_id} {size}] parse={row['parse_rate']:.3f} "
          f"feasible={row['feasible_rate']:.3f} avg_gen={row['avg_gen_s']:.2f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="6x5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--blanks", type=int, default=6)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    a = ap.parse_args()
    run_slm(a.size, a.n, a.seed, a.blanks, a.model)
