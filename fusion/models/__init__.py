from __future__ import annotations

from typing import Any

from .module_screening import (
    PlainConditionalCSPL1ScreeningNet,
    PlainE1DualBandDiffNet,
    PlainE1DualBandPredNet,
    PlainE1DualBandPredSourceVNet,
    PlainOriginalRFGML2FusionNet,
)
from .plain_fusion import PlainFusionNet


MODEL_REGISTRY = {
    "plain_fusion": PlainFusionNet,
    "plain_conditional_csp_l1_fusion": PlainConditionalCSPL1ScreeningNet,
    "plain_e1_dualband_diff_l2": PlainE1DualBandDiffNet,
    "plain_e1_dualband_pred_l2": PlainE1DualBandPredNet,
    "plain_e1_dualband_pred_sourcev_l2": PlainE1DualBandPredSourceVNet,
    "plain_original_rfgm_l2_fusion": PlainOriginalRFGML2FusionNet,
}


def build_model(name: str, **kwargs: Any):
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "PlainFusionNet",
    "PlainConditionalCSPL1ScreeningNet",
    "PlainE1DualBandDiffNet",
    "PlainE1DualBandPredNet",
    "PlainE1DualBandPredSourceVNet",
    "PlainOriginalRFGML2FusionNet",
    "build_model",
    "MODEL_REGISTRY",
]
