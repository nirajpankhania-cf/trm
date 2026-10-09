"""Shared training routine for experiments (returns EMA-evaluated model)."""
from __future__ import annotations

import time

import numpy as np
import torch

from trm_industry.model import TRM, TRMConfig
from trm_industry.train import EMA, _supervised_forward_loss, evaluate_grid
from trm_industry.utils import count_params, human, set_seed

RESULTS = "results"
CKPTS = "results/ckpts"


def train(task, spec, tr, te, *, recursive, hidden, n_latent, T, n_sup,
          epochs, lr, blanks, seed=0, device="cpu", threads=4, q_weight=0.5,
          eval_every=0, verbose=True):
    """Train one model and return (model_with_ema_weights, history, final_metrics)."""
    torch.set_num_threads(threads)
    set_seed(seed)
    dev = torch.device(device)
    tr_in, tr_tg, tr_pz, tr_sol = tr
    te_in, te_pz, te_sol = te
    cfg = TRMConfig(vocab_size=spec.vocab_size, num_classes=spec.num_classes,
                    seq_len=spec.seq_len, hidden=hidden, n_layers=2,
                    n_latent=n_latent, T=T, n_sup=n_sup, recursive=recursive,
                    core_type="mlp")
    model = TRM(cfg).to(dev)
    nparams = count_params(model)
    inp, tgt = tr_in.to(dev), tr_tg.to(dev)
    mask = (inp == 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema = EMA(model, 0.995)
    bs = 512
    n = inp.shape[0]
    history = []
    gshape = spec.shape if hasattr(spec, "shape") else (spec.n, spec.n)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        el, nb = 0.0, 0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            loss = _supervised_forward_loss(model, inp[idx], tgt[idx], mask[idx], q_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model)
            el += float(loss.detach()); nb += 1
        sched.step()
        history.append(el / max(1, nb))
        if eval_every and (ep % eval_every == 0 or ep == epochs - 1):
            bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            m = evaluate_grid(model, te_in, te_pz, te_sol, spec, task.verify, dev,
                              assemble=task.assemble, grid_shape=gshape)
            model.load_state_dict(bak)
            if verbose:
                print(f"  ep {ep:3d} loss {history[-1]:.3f} "
                      f"exact {m['exact_match']:.3f} valid {m['valid_solve']:.3f}", flush=True)
    train_time = time.perf_counter() - t0
    ema.copy_to(model)  # use EMA weights for everything downstream
    metrics = evaluate_grid(model, te_in, te_pz, te_sol, spec, task.verify, dev,
                            assemble=task.assemble, grid_shape=gshape)
    metrics.update({"params": int(nparams), "params_h": human(nparams),
                    "train_time_s": round(train_time, 1), "recursive": recursive,
                    "n_latent": n_latent, "T": T, "n_sup": n_sup, "epochs": epochs,
                    "blanks": blanks, "final_loss": round(history[-1], 4)})
    return model, history, metrics
