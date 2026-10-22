from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class _ConditionalDirectionalCompensation(nn.Module):
    """Condition donor frequency bands and spatial injection on both modalities."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        context_hidden = max(channels // 2, 16)
        self.groups = groups
        self.group_size = channels // groups

        # Keep the value path identical to the original CSP for a controlled
        # comparison.  Normalization is used only by decision paths below.
        self.transfer = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        )
        self.local = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        )

        # The receiver selects where the donor is summarized.  This is a
        # context descriptor, not the final spatial injection gate.
        self.receiver_key = nn.Conv2d(channels, 1, kernel_size=1)
        self.band_context = nn.Sequential(
            nn.Conv2d(channels * 3, context_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(context_hidden, groups * 2, kernel_size=1),
        )
        # Start with the original static 0.5 band gains.  The conditional
        # residual learns only after evidence supports a sample-wise change.
        nn.init.zeros_(self.band_context[-1].weight)
        nn.init.zeros_(self.band_context[-1].bias)
        self.band_logits = nn.Parameter(torch.zeros(groups, 2))

        # Shared evidence extractor with independent sigmoid decisions for the
        # spectral and local residuals.  Neither branch is forced to be used.
        self.gate_features = nn.Sequential(
            nn.Conv2d(5, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gate_out = nn.Conv2d(hidden, 2, kernel_size=1)

    @staticmethod
    def _evidence(x: Tensor) -> tuple[Tensor, Tensor]:
        energy = x.abs().mean(dim=1, keepdim=True)
        detail = (x - F.avg_pool2d(x, 3, stride=1, padding=1)).abs().mean(dim=1, keepdim=True)
        return energy, detail

    def _conditional_band_gains(self, mapped: Tensor, current: Tensor) -> Tensor:
        batch, channels, height, width = mapped.shape
        receiver_weights = self.receiver_key(current).flatten(2).softmax(dim=-1)
        donor_context = torch.bmm(mapped.flatten(2), receiver_weights.transpose(1, 2)).view(batch, channels, 1, 1)
        descriptor = torch.cat(
            [
                F.adaptive_avg_pool2d(mapped, 1),
                F.adaptive_avg_pool2d(current, 1),
                donor_context,
            ],
            dim=1,
        )
        conditional_logits = self.band_context(descriptor).view(batch, self.groups, 2)
        return torch.sigmoid(self.band_logits.unsqueeze(0) + conditional_logits)

    def forward(
        self,
        donor: Tensor,
        current: Tensor,
        low_mask: Tensor,
        mid_mask: Tensor,
        branch_mode: str | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        mapped = self.transfer(donor)
        donor_energy, donor_detail = self._evidence(mapped)
        current_energy, current_detail = self._evidence(current)
        evidence = torch.cat(
            [donor_energy, current_energy, donor_detail, current_detail, (donor_energy - current_energy).abs()],
            dim=1,
        )
        gates = torch.sigmoid(self.gate_out(self.gate_features(evidence)))
        spectral_gate, local_gate = gates[:, 0:1], gates[:, 1:2]

        # FFT moment subtraction/filtering is kept in FP32 under outer AMP.
        with torch.autocast(device_type=mapped.device.type, enabled=False):
            mapped_fp32 = mapped.float()
            spectrum = torch.fft.rfft2(mapped_fp32, norm="ortho")
            batch, channels, height, width_frequency = spectrum.shape
            grouped = spectrum.view(batch, self.groups, self.group_size, height, width_frequency)
            band_gains = self._conditional_band_gains(mapped_fp32, current.float())
            gains = band_gains.view(batch, self.groups, 1, 2, 1, 1)
            masks = torch.stack([low_mask.float(), mid_mask.float()], dim=3)
            filtered = (grouped.unsqueeze(3) * gains * masks).sum(dim=3).reshape(
                batch, channels, height, width_frequency
            )
            spectral_residual = torch.fft.irfft2(filtered, s=current.shape[-2:], norm="ortho")
        spectral_residual = spectral_residual.to(current.dtype)
        local_residual = self.local(torch.cat([mapped, current, (mapped - current).abs()], dim=1))

        weighted_spectral = spectral_gate * spectral_residual
        weighted_local = local_gate * local_residual
        if branch_mode == "spectral_only":
            weighted_local = weighted_local * 0.0
        elif branch_mode == "local_only":
            weighted_spectral = weighted_spectral * 0.0
        residual = weighted_spectral + weighted_local
        statistics = {
            "band_gain": band_gains.detach(),
            "spectral_gate": spectral_gate.detach(),
            "local_gate": local_gate.detach(),
            "weighted_spectral": weighted_spectral.detach(),
            "weighted_local": weighted_local.detach(),
        }
        return residual, statistics


class ConditionalCrossModalSpectralPrior(nn.Module):
    """Bidirectional conditional spectral/local compensation at the original CSP location."""

    def __init__(
        self,
        channels: int,
        groups: int = 4,
        alpha_max: float = 0.50,
        alpha_init: float = 0.20,
        low_cutoff: float = 0.12,
        mid_cutoff: float = 0.32,
        transition: float = 0.04,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups != 0:
            raise ValueError("channels must be positive and divisible by groups.")
        if not 0.0 < alpha_init < alpha_max < 1.0:
            raise ValueError("Require 0 < alpha_init < alpha_max < 1.")
        if not 0.0 < low_cutoff < mid_cutoff:
            raise ValueError("Require 0 < low_cutoff < mid_cutoff.")
        if transition < 0.0:
            raise ValueError("transition must be non-negative.")

        self.alpha_max = float(alpha_max)
        self.low_cutoff = float(low_cutoff)
        self.mid_cutoff = float(mid_cutoff)
        self.transition = float(transition)
        self.eps = float(eps)
        theta = _logit(alpha_init / alpha_max)
        self.alpha_ir_theta = nn.Parameter(torch.tensor(theta))
        self.alpha_vis_theta = nn.Parameter(torch.tensor(theta))
        self.vis_to_ir = _ConditionalDirectionalCompensation(channels, groups)
        self.ir_to_vis = _ConditionalDirectionalCompensation(channels, groups)
        self.diagnostic_mode: str | None = None
        self.last_statistics: dict[str, Tensor] = {}

    def _alpha_ir(self) -> Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_ir_theta)

    def _alpha_vis(self) -> Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_vis_theta)

    def _smooth_lowpass(self, radius: Tensor, cutoff: float) -> Tensor:
        if self.transition == 0.0:
            return (radius <= cutoff).to(radius.dtype)
        start = max(cutoff - self.transition * 0.5, 0.0)
        end = cutoff + self.transition * 0.5
        phase = ((radius - start) / max(end - start, self.eps)).clamp(0.0, 1.0)
        tapered = 0.5 * (1.0 + torch.cos(math.pi * phase))
        return torch.where(radius <= start, torch.ones_like(radius), torch.where(radius >= end, torch.zeros_like(radius), tapered))

    def _frequency_masks(self, height: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
        fy = torch.fft.fftfreq(height, device=device).view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device).view(1, width // 2 + 1)
        radius = torch.sqrt(fy.square() + fx.square())
        low = self._smooth_lowpass(radius, self.low_cutoff)
        low_mid = self._smooth_lowpass(radius, self.mid_cutoff)
        mid = (low_mid - low).clamp_min(0.0)
        low[0, 0] = 0.0
        mid[0, 0] = 0.0
        return low.view(1, 1, 1, height, -1), mid.view(1, 1, 1, height, -1)

    def _ratio(self, value: Tensor, reference: Tensor) -> Tensor:
        value_norm = value.flatten(1).norm(dim=1)
        reference_norm = reference.flatten(1).norm(dim=1).clamp_min(self.eps)
        return (value_norm / reference_norm).detach()

    def forward(self, vis: Tensor, ir: Tensor) -> tuple[Tensor, Tensor]:
        if vis.shape != ir.shape or vis.ndim != 4:
            raise ValueError(f"Expected matching [B,C,H,W] features, got {tuple(vis.shape)} and {tuple(ir.shape)}.")
        low, mid = self._frequency_masks(*vis.shape[-2:], vis.device)
        branch_mode = self.diagnostic_mode if self.diagnostic_mode in {"spectral_only", "local_only"} else None

        # Both directions read the same original pair; neither direction sees
        # the other direction's already-updated result.
        ir_residual, ir_stats = self.vis_to_ir(vis, ir, low, mid, branch_mode)
        vis_residual, vis_stats = self.ir_to_vis(ir, vis, low, mid, branch_mode)
        alpha_ir = self._alpha_ir().to(ir)
        alpha_vis = self._alpha_vis().to(vis)
        if self.diagnostic_mode == "bypass":
            alpha_ir = alpha_ir * 0.0
            alpha_vis = alpha_vis * 0.0
        elif self.diagnostic_mode == "vis_to_ir":
            alpha_vis = alpha_vis * 0.0
        elif self.diagnostic_mode == "ir_to_vis":
            alpha_ir = alpha_ir * 0.0

        ir_delta = alpha_ir * ir_residual
        vis_delta = alpha_vis * vis_residual
        self.last_statistics = {
            "ir_ratio": self._ratio(ir_delta, ir),
            "vis_ratio": self._ratio(vis_delta, vis),
            "alpha_ir": alpha_ir.detach().reshape(1),
            "alpha_vis": alpha_vis.detach().reshape(1),
            "band_gain_ir": ir_stats["band_gain"],
            "band_gain_vis": vis_stats["band_gain"],
            "spectral_gate_ir": ir_stats["spectral_gate"],
            "spectral_gate_vis": vis_stats["spectral_gate"],
            "local_gate_ir": ir_stats["local_gate"],
            "local_gate_vis": vis_stats["local_gate"],
            "spectral_ratio_ir": self._ratio(alpha_ir * ir_stats["weighted_spectral"], ir),
            "spectral_ratio_vis": self._ratio(alpha_vis * vis_stats["weighted_spectral"], vis),
            "local_ratio_ir": self._ratio(alpha_ir * ir_stats["weighted_local"], ir),
            "local_ratio_vis": self._ratio(alpha_vis * vis_stats["weighted_local"], vis),
        }
        return vis + vis_delta, ir + ir_delta


__all__ = ["ConditionalCrossModalSpectralPrior"]
