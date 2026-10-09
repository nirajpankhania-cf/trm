"""Save/load trained models with their config."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .model import TRM, TRMConfig


def save(path: str, model: TRM, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cfg": asdict(model.cfg),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }, path)


def load(path: str, device) -> tuple[TRM, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = TRMConfig(**blob["cfg"])
    model = TRM(cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("extra", {})


def write_json(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
