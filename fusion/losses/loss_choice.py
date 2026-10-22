from __future__ import annotations

from math import exp, isfinite

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .base_loss import _grad_components, _to_gray


def _check_weight(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _absolute_gradient_loss(fused: Tensor, vis: Tensor, ir: Tensor) -> Tensor:
    gx_v, gy_v = _grad_components(vis)
    gx_i, gy_i = _grad_components(ir)
    gx_f, gy_f = _grad_components(fused)
    target_x = torch.maximum(gx_v.abs(), gx_i.abs())
    target_y = torch.maximum(gy_v.abs(), gy_i.abs())
    return F.l1_loss(gx_f.abs(), target_x) + F.l1_loss(gy_f.abs(), target_y)


def _gaussian_window(window_size: int, channels: int, x: Tensor) -> Tensor:
    # Both source implementations center even windows at window_size // 2.
    values = [exp(-((index - window_size // 2) ** 2) / (2.0 * 1.5**2)) for index in range(window_size)]
    one_d = x.new_tensor(values)
    one_d = one_d / one_d.sum()
    window = (one_d[:, None] @ one_d[None, :]).view(1, 1, window_size, window_size)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def _controlfusion_ssim_loss(x: Tensor, y: Tensor, window_size: int = 48) -> Tensor:
    """ControlFusion SSIM: Gaussian sigma=1.5, 48x48 and valid convolution."""
    if x.shape != y.shape:
        raise ValueError(f"ControlFusion SSIM inputs differ: {tuple(x.shape)} != {tuple(y.shape)}")
    if min(x.shape[-2:]) < window_size:
        raise ValueError(f"ControlFusion SSIM needs images at least {window_size}x{window_size}.")
    channels = x.shape[1]
    window = _gaussian_window(window_size, channels, x)
    mu_x = F.conv2d(x, window, groups=channels)
    mu_y = F.conv2d(y, window, groups=channels)
    mu_x_sq, mu_y_sq, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    var_x = F.conv2d(x.square(), window, groups=channels) - mu_x_sq
    var_y = F.conv2d(y.square(), window, groups=channels) - mu_y_sq
    cov_xy = F.conv2d(x * y, window, groups=channels) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2.0 * mu_xy + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (var_x + var_y + c2)
    return 1.0 - (numerator / denominator).mean()


def _vis_chroma_rgb(fused_y: Tensor, vis_rgb: Tensor) -> Tensor:
    """Differentiable YCbCr inverse using fused Y and visible-image chroma."""
    r, g, b = vis_rgb[:, 0:1], vis_rgb[:, 1:2], vis_rgb[:, 2:3]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    fused_rgb = torch.cat(
        [
            fused_y + 1.402 * cr,
            fused_y - 0.344136 * cb - 0.714136 * cr,
            fused_y + 1.772 * cb,
        ],
        dim=1,
    )
    return fused_rgb.clamp(0.0, 1.0)


def _color_loss(vis_rgb: Tensor, fused_rgb: Tensor) -> Tensor:
    def chroma(x: Tensor) -> tuple[Tensor, Tensor]:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        return cb, cr

    cb_v, cr_v = chroma(vis_rgb)
    cb_f, cr_f = chroma(fused_rgb)
    return F.l1_loss(cb_f, cb_v) + F.l1_loss(cr_f, cr_v)


class ControlFusionC0Loss(nn.Module):
    """Official ControlFusion fusion loss with a fixed Y-to-RGB adapter."""

    def __init__(
        self,
        intensity_weight: float = 8.0,
        gradient_weight: float = 10.0,
        ssim_weight: float = 1.0,
        color_weight: float = 12.0,
    ) -> None:
        super().__init__()
        self.intensity_weight = _check_weight(intensity_weight, "intensity_weight")
        self.gradient_weight = _check_weight(gradient_weight, "gradient_weight")
        self.ssim_weight = _check_weight(ssim_weight, "ssim_weight")
        self.color_weight = _check_weight(color_weight, "color_weight")

    def compute_terms(self, fused_y: Tensor, vis: Tensor, ir: Tensor) -> dict[str, Tensor]:
        with torch.autocast(device_type=fused_y.device.type, enabled=False):
            fused_y = _to_gray(fused_y.float())
            vis_rgb = vis.float()
            ir_gray = _to_gray(ir.float())
            fused_rgb = _vis_chroma_rgb(fused_y, vis_rgb)
            vis_gray = _to_gray(vis_rgb)
            fused_gray = _to_gray(fused_rgb)

            # This intentionally uses ControlFusion's RGB-channel mean mask.
            mask = ir_gray > vis_rgb.mean(dim=1, keepdim=True)
            target_rgb = torch.where(mask, ir_gray, vis_rgb)
            intensity = F.l1_loss(fused_rgb, target_rgb)
            gradient = _absolute_gradient_loss(fused_gray, vis_gray, ir_gray)
            ssim_vis = _controlfusion_ssim_loss(vis_rgb, fused_rgb)
            ssim_ir = _controlfusion_ssim_loss(ir_gray, fused_gray)
            color = _color_loss(vis_rgb, fused_rgb)
        return {
            "intensity": intensity,
            "gradient": gradient,
            "ssim": ssim_vis + ssim_ir,
            "ssim_vis_loss": ssim_vis,
            "ssim_ir_loss": ssim_ir,
            "color": color,
        }

    def forward(self, fused_y: Tensor, vis: Tensor, ir: Tensor) -> dict[str, Tensor]:
        terms = self.compute_terms(fused_y, vis, ir)
        weighted_intensity = self.intensity_weight * terms["intensity"]
        weighted_gradient = self.gradient_weight * terms["gradient"]
        weighted_ssim = self.ssim_weight * terms["ssim"]
        weighted_color = self.color_weight * terms["color"]
        total = weighted_intensity + weighted_gradient + weighted_ssim + weighted_color
        return {
            "loss": total,
            **{key: value.detach() for key, value in terms.items()},
            "weighted_intensity": weighted_intensity.detach(),
            "weighted_gradient": weighted_gradient.detach(),
            "weighted_ssim": weighted_ssim.detach(),
            "weighted_color": weighted_color.detach(),
        }


__all__ = ["ControlFusionC0Loss"]
