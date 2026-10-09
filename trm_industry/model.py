"""TRM-style tiny recursive reasoning model.

Faithful-in-spirit reproduction of the mechanics described for the Tiny
Recursive Model (Jolicoeur-Martineau, arXiv:2510.04871):

* a single small network (2-layer transformer by default) is *recursed*;
* two latent states are carried: ``y`` (current answer) and ``z`` (reasoning);
* latent recursion ``z <- net(x, y, z)`` is repeated ``n`` times, then the
  answer is updated ``y <- net(x, y, z)``;
* "deep recursion" runs ``T`` such blocks per supervision step, with gradient
  flowing only through the final block (truncated BPTT);
* deep supervision applies the loss at each of up to ``N_sup`` steps with the
  carried state detached between steps;
* an EMA copy of the weights is kept for evaluation (handled in train.py).

The same class also supports a non-recursive one-pass baseline so we can
reproduce the paper's central ablation (recursion + deep supervision is the
driver, not depth/params alone).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TRMConfig:
    vocab_size: int          # distinct input tokens (incl. blank=0)
    num_classes: int         # output classes per cell
    seq_len: int             # flattened grid length L
    hidden: int = 128
    n_layers: int = 2        # transformer layers inside the single core net
    n_heads: int = 4
    ff_mult: int = 2
    n_latent: int = 6        # n: latent recursions per block
    T: int = 3               # deep recursions per supervision step
    n_sup: int = 8           # deep-supervision steps
    dropout: float = 0.0
    recursive: bool = True   # False -> one-pass baseline
    core_type: str = "mlp"   # "mlp" (TRM-MLP, fast/best on fixed grids) or "attn"


class _Block(nn.Module):
    """Pre-norm transformer encoder block (self-attention + SwiGLU-ish MLP)."""

    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden)
        self.attn = nn.MultiheadAttention(
            cfg.hidden, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.hidden)
        hidden_ff = cfg.hidden * cfg.ff_mult
        self.w_in = nn.Linear(cfg.hidden, hidden_ff)
        self.w_gate = nn.Linear(cfg.hidden, hidden_ff)
        self.w_out = nn.Linear(hidden_ff, cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm1(h)
        a, _ = self.attn(x, x, x, need_weights=False)
        h = h + self.drop(a)
        x = self.norm2(h)
        x = self.w_out(torch.nn.functional.silu(self.w_gate(x)) * self.w_in(x))
        return h + self.drop(x)


class _MixerBlock(nn.Module):
    """MLP-mixer block: token-mixing (across L) + channel-mixing (across H).

    This is the paper's TRM-MLP variant, reported strongest on small fixed
    grids (e.g. Sudoku). It uses only matmuls, so it is far faster than
    attention for tiny sequences on CPU/MPS.
    """

    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden)
        tok_hidden = max(cfg.seq_len, cfg.seq_len * cfg.ff_mult // 1)
        self.tok_in = nn.Linear(cfg.seq_len, tok_hidden)
        self.tok_out = nn.Linear(tok_hidden, cfg.seq_len)
        self.norm2 = nn.LayerNorm(cfg.hidden)
        ch_hidden = cfg.hidden * cfg.ff_mult
        self.ch_in = nn.Linear(cfg.hidden, ch_hidden)
        self.ch_gate = nn.Linear(cfg.hidden, ch_hidden)
        self.ch_out = nn.Linear(ch_hidden, cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # token mixing
        x = self.norm1(h).transpose(1, 2)            # [B,H,L]
        x = self.tok_out(torch.nn.functional.silu(self.tok_in(x)))
        h = h + self.drop(x.transpose(1, 2))
        # channel mixing (SwiGLU)
        x = self.norm2(h)
        x = self.ch_out(torch.nn.functional.silu(self.ch_gate(x)) * self.ch_in(x))
        return h + self.drop(x)


class Core(nn.Module):
    """The single network that gets recursed."""

    def __init__(self, cfg: TRMConfig):
        super().__init__()
        block_cls = _MixerBlock if cfg.core_type == "mlp" else _Block
        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)


class TRM(nn.Module):
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.hidden))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.core = Core(cfg)
        self.head = nn.Linear(cfg.hidden, cfg.num_classes)
        # Q/halt head: predicts whether current answer is fully correct.
        self.q_head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 1)
        )

    # ---- embeddings -----------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(x) + self.pos_emb

    def init_state(self, x_emb: torch.Tensor):
        y = torch.zeros_like(x_emb)
        z = torch.zeros_like(x_emb)
        return y, z

    # ---- recursion primitive -------------------------------------------
    def one_block(self, x_emb, y, z, noise_sigma: float = 0.0):
        """One deep-recursion block: n latent updates then an answer update."""
        for _ in range(self.cfg.n_latent):
            z = self.core(x_emb + y + z)
            if noise_sigma > 0:
                z = z + noise_sigma * torch.randn_like(z)
        y = self.core(x_emb + y + z)
        return y, z

    def deep_recursion(self, x_emb, y, z, use_grad: bool, noise_sigma: float = 0.0):
        """T blocks; gradient only through the last (truncated BPTT)."""
        T = self.cfg.T
        with torch.no_grad():
            for _ in range(max(0, T - 1)):
                y, z = self.one_block(x_emb, y, z, noise_sigma)
        if use_grad:
            y, z = self.one_block(x_emb, y, z, noise_sigma)
        else:
            with torch.no_grad():
                y, z = self.one_block(x_emb, y, z, noise_sigma)
        return y, z

    # ---- heads ----------------------------------------------------------
    def logits(self, y):
        return self.head(y)

    def q_logit(self, y):
        # pool over sequence then project to a single correctness logit
        return self.q_head(y.mean(dim=1)).squeeze(-1)

    # ---- one-pass baseline ---------------------------------------------
    def forward_onepass(self, x):
        h = self.core(self.embed(x))
        return self.logits(h), self.q_logit(h)

    @torch.no_grad()
    def solve(self, x, noise_sigma: float = 0.0, n_sup: int | None = None):
        """Full inference. Returns (logits, q_prob) at the final step.

        If ``noise_sigma>0`` this realises PTRM-style stochastic rollouts.
        """
        if not self.cfg.recursive:
            lg, q = self.forward_onepass(x)
            return lg, torch.sigmoid(q)
        steps = n_sup if n_sup is not None else self.cfg.n_sup
        x_emb = self.embed(x)
        y, z = self.init_state(x_emb)
        for _ in range(steps):
            y, z = self.deep_recursion(x_emb, y, z, use_grad=False, noise_sigma=noise_sigma)
        return self.logits(y), torch.sigmoid(self.q_logit(y))
