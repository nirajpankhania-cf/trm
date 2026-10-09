"""Training (deep supervision + full backprop + EMA) and evaluation helpers.

Generic over the task: a task provides tensors of inputs/targets plus a
verifier closure, so the same loop trains Sudoku and rostering.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .model import TRM


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow)


@dataclass
class TrainCfg:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1.0
    q_weight: float = 0.5
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    log_every: int = 20
    eval_blanks_as_mask: bool = True  # loss only on blank cells


def _supervised_forward_loss(model: TRM, x, target, blank_mask, q_weight: float):
    """Run deep supervision and return summed loss over steps."""
    cfg = model.cfg
    if not cfg.recursive:
        logits, q = model.forward_onepass(x)
        ce = _masked_ce(logits, target, blank_mask)
        correct = _cellwise_correct(logits, target)
        qt = correct.all(dim=1).float()
        qloss = F.binary_cross_entropy_with_logits(q, qt)
        return ce + q_weight * qloss

    x_emb = model.embed(x)
    y, z = model.init_state(x_emb)
    total = 0.0
    for _ in range(cfg.n_sup):
        y, z = model.deep_recursion(x_emb, y, z, use_grad=True)
        logits = model.logits(y)
        ce = _masked_ce(logits, target, blank_mask)
        with torch.no_grad():
            correct = _cellwise_correct(logits, target)
            qt = correct.all(dim=1).float()
        qloss = F.binary_cross_entropy_with_logits(model.q_logit(y), qt)
        total = total + ce + q_weight * qloss
        y, z = y.detach(), z.detach()
    return total


def _masked_ce(logits, target, blank_mask):
    # logits: [B,L,C], target: [B,L], blank_mask: [B,L] bool
    B, L, C = logits.shape
    ce = F.cross_entropy(logits.reshape(B * L, C), target.reshape(B * L), reduction="none")
    ce = ce.reshape(B, L)
    if blank_mask is not None:
        m = blank_mask.float()
        return (ce * m).sum() / m.sum().clamp_min(1.0)
    return ce.mean()


def _cellwise_correct(logits, target):
    return logits.argmax(dim=-1) == target


def train_model(model: TRM, data, device, tcfg: TrainCfg, verbose: bool = True):
    """data = dict with inputs[B,L], targets[B,L]. Returns (ema, history)."""
    model.to(device)
    inputs = data["inputs"].to(device)
    targets = data["targets"].to(device)
    blank_mask = (inputs == 0) if tcfg.eval_blanks_as_mask else None

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    ema = EMA(model, decay=tcfg.ema_decay)

    n = inputs.shape[0]
    history = []
    for epoch in range(tcfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        nb = 0
        for s in range(0, n, tcfg.batch_size):
            idx = perm[s:s + tcfg.batch_size]
            xb, tb = inputs[idx], targets[idx]
            mb = blank_mask[idx] if blank_mask is not None else None
            opt.zero_grad(set_to_none=True)
            loss = _supervised_forward_loss(model, xb, tb, mb, tcfg.q_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            ema.update(model)
            epoch_loss += float(loss.detach())
            nb += 1
        sched.step()
        history.append(epoch_loss / max(1, nb))
        if verbose and (epoch % tcfg.log_every == 0 or epoch == tcfg.epochs - 1):
            print(f"  epoch {epoch:4d}  loss {history[-1]:.4f}")
    return ema, history


@torch.no_grad()
def decode(model: TRM, x, device, noise_sigma: float = 0.0, n_sup: int | None = None):
    """Return predicted class grids [B,L] (0-indexed) and q-prob [B]."""
    model.eval()
    x = x.to(device)
    logits, q = model.solve(x, noise_sigma=noise_sigma, n_sup=n_sup)
    return logits.argmax(dim=-1), q, logits


def _sudoku_assemble(puz, pred_classes_grid):
    """Default assemble for Sudoku: answer in digit space (1..n)."""
    return np.where(puz > 0, puz, pred_classes_grid + 1)


@torch.no_grad()
def evaluate_grid(model, x, puzzles, sols, spec, verify_fn, device, n_sup=None,
                  assemble=_sudoku_assemble, grid_shape=None):
    """Exact-match and valid-solve rates on a grid task (deterministic)."""
    preds, _, _ = decode(model, x, device, noise_sigma=0.0, n_sup=n_sup)
    preds = preds.cpu().numpy()
    B = preds.shape[0]
    shape = grid_shape if grid_shape is not None else (spec.n, spec.n)
    exact = 0
    valid = 0
    for i in range(B):
        puz = puzzles[i]
        pred_grid = preds[i].reshape(shape)
        ans = assemble(puz, pred_grid)
        if (ans == sols[i]).all():
            exact += 1
        if verify_fn(ans, puz, spec):
            valid += 1
    return {"exact_match": exact / B, "valid_solve": valid / B, "n": B}
