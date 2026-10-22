from __future__ import annotations

import torch
from torch import Tensor

from .conditional_spectral_prior import ConditionalCrossModalSpectralPrior
from .dual_band_fusion import DualBandFusion
from .encoder_fusion_rfgm import EncoderFusionRFGMRefine
from .plain_fusion import PlainFusionNet


def _identity_projection(layer: torch.nn.Conv2d) -> None:
    with torch.no_grad():
        layer.weight.zero_()
        diagonal = torch.arange(min(layer.in_channels, layer.out_channels))
        layer.weight[diagonal, diagonal, 0, 0] = 1.0
        if layer.bias is not None:
            layer.bias.zero_()


def _build_rfgm(channels: int) -> EncoderFusionRFGMRefine:
    module = EncoderFusionRFGMRefine(
        current_channels=channels,
        prior_channels=channels,
        alpha_init=0.10,
        alpha_max=0.20,
        top_k=4,
        topk_temperature=0.25,
        amp_scale_init=0.015,
        amp_scale_max=0.05,
        phase_scale_init=0.004,
        phase_scale_max=0.015,
        beta_init=0.08,
        beta_max=0.18,
    )
    _identity_projection(module.res_proj)
    module.last_residual_ratio = torch.zeros(1)
    return module


def _refine(module: EncoderFusionRFGMRefine, current: Tensor, prior: Tensor) -> Tensor:
    output = module(current, prior)
    with torch.no_grad():
        numerator = (output - current).float().flatten(1).norm(dim=1)
        denominator = current.float().flatten(1).norm(dim=1).clamp_min(1e-6)
        module.last_residual_ratio = numerator / denominator
    return output


class PlainConditionalCSPL1ScreeningNet(PlainFusionNet):
    """E1 and the base class for future experiments: CSP before fuse1."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.csp1 = ConditionalCrossModalSpectralPrior(
            self.fuse1.op[0].out_channels, alpha_init=0.40, alpha_max=0.50
        )

    def prepare_l1(self, vis_l1: Tensor, ir_l1: Tensor) -> tuple[Tensor, Tensor]:
        return self.csp1(vis_l1, ir_l1)


class PlainOriginalRFGML2FusionNet(PlainFusionNet):
    """E3: RFGM after merge2 and before dec2."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rfgm2 = _build_rfgm(self.fuse2.op[0].out_channels)

    def refine_l2(self, current: Tensor, prior: Tensor) -> Tensor:
        return _refine(self.rfgm2, current, prior)


class PlainE1DualBandDiffNet(PlainConditionalCSPL1ScreeningNet):
    """F1: E1 with direct L2 dual-band fusion and difference messages."""

    kv_mode = "difference"
    source_values = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        channels = self.fuse2.op[0].out_channels
        self.fuse2 = DualBandFusion(
            channels,
            mode=self.kv_mode,
            source_values=self.source_values,
        )

    def pop_prediction_loss(self) -> Tensor | None:
        return self.fuse2.pop_prediction_loss()


class PlainE1DualBandPredNet(PlainE1DualBandDiffNet):
    """F2: F1 structure with supervised prediction-residual messages."""

    kv_mode = "prediction"


class PlainE1DualBandPredSourceVNet(PlainE1DualBandPredNet):
    """F3: prediction residuals form K; original auxiliary lows form V."""

    source_values = True


__all__ = [
    "PlainConditionalCSPL1ScreeningNet",
    "PlainE1DualBandDiffNet",
    "PlainE1DualBandPredNet",
    "PlainE1DualBandPredSourceVNet",
    "PlainOriginalRFGML2FusionNet",
]
