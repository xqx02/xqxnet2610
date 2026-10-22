from __future__ import annotations

from typing import Any

from .loss_choice import ControlFusionC0Loss


LOSS_REGISTRY = {
    "controlfusion_c0_yadapter": ControlFusionC0Loss,
}


def build_loss(name: str, **kwargs: Any):
    if name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{name}'. Available losses: {available}")
    return LOSS_REGISTRY[name](**kwargs)


__all__ = [
    "ControlFusionC0Loss",
    "build_loss",
    "LOSS_REGISTRY",
]

