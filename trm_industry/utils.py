"""Shared utilities: device selection, seeding, timing, small helpers."""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device.

    Order of preference: explicit arg -> MPS (Apple) -> CUDA -> CPU.
    """
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timer():
    """Context manager returning elapsed seconds via the yielded callable."""
    start = time.perf_counter()
    elapsed = {}

    def read() -> float:
        return elapsed.get("t", time.perf_counter() - start)

    try:
        yield read
    finally:
        elapsed["t"] = time.perf_counter() - start


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def human(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"
