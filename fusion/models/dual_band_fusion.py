from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LearnableFDM(nn.Module):
    """Learnable local low/high decomposition adapted from Fusion's FDM.

    Kernel generation and parameter names follow ``Fusion/model/_fdafm_full.py``.
    A sample-folded depthwise convolution replaces its explicit unfold tensor.
    """

    def __init__(self, channels: int, kernel_size: int = 3, group: int = 8) -> None:
        super().__init__()
        if channels % group != 0:
            group = max(math.gcd(channels, group), 1)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.channels = channels
        self.kernel_size = kernel_size
        self.group = group
        self.kernel_area = kernel_size * kernel_size
        self.filter_gen = nn.Conv2d(
            channels, group * self.kernel_area, kernel_size=1, bias=False
        )
        self.norm = nn.GroupNorm(group, group * self.kernel_area)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        nn.init.zeros_(self.filter_gen.weight)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = x.shape
        kernel = self.norm(self.filter_gen(self.pool(x)))
        kernel = kernel.reshape(batch, self.group, self.kernel_area)
        kernel = torch.softmax(kernel, dim=-1)

        weight = kernel.repeat_interleave(channels // self.group, dim=1)
        weight = weight.reshape(
            batch * channels, 1, self.kernel_size, self.kernel_size
        )
        padded = self.pad(x)
        folded = padded.reshape(
            1, batch * channels, padded.shape[-2], padded.shape[-1]
        )
        low = F.conv2d(folded, weight, groups=batch * channels)
        low = low.reshape(batch, channels, height, width)
        return low, x - low


class ChannelCrossAttention(nn.Module):
    """VideoFusion channel cross-attention core without its output projection."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.kv = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.q_dwconv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.kv_dwconv = nn.Conv2d(
            2 * channels,
            2 * channels,
            3,
            padding=1,
            groups=2 * channels,
            bias=False,
        )

    def forward(
        self,
        query_source: Tensor,
        message: Tensor,
        value_source: Tensor | None = None,
    ) -> Tensor:
        batch, channels, height, width = query_source.shape
        q = self.q_dwconv(self.q(query_source))
        if value_source is None:
            kv = self.kv(message)
        else:
            key = F.conv2d(message, self.kv.weight[:channels])
            value = F.conv2d(value_source, self.kv.weight[channels:])
            kv = torch.cat([key, value], dim=1)
        k, v = self.kv_dwconv(kv).chunk(2, dim=1)
        shape = (batch, self.heads, channels // self.heads, height * width)
        q = F.normalize(q.reshape(shape), dim=-1)
        k = F.normalize(k.reshape(shape), dim=-1)
        v = v.reshape(shape)
        attention = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return (attention @ v).reshape(batch, channels, height, width)


class SplitScalePredictor(nn.Module):
    """Full channel mapping with a narrow local/dilated context branch."""

    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        if hidden <= 0 or hidden % 2:
            raise ValueError("hidden must be a positive even integer")
        self.channels = channels
        self.hidden = hidden
        self.in_proj = nn.Conv2d(channels, channels + hidden, 1, bias=False)
        half = hidden // 2
        self.local = nn.Conv2d(
            half, half, 3, padding=1, groups=half, bias=False
        )
        self.wide = nn.Conv2d(
            half, half, 3, padding=2, dilation=2, groups=half, bias=False
        )
        self.out_proj = nn.Conv2d(hidden, channels, 1, bias=False)

        nn.init.dirac_(self.in_proj.weight[:channels])
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        linear, context = self.in_proj(x).split(
            [self.channels, self.hidden], dim=1
        )
        local, wide = context.chunk(2, dim=1)
        context = torch.cat(
            [F.gelu(self.local(local)), F.gelu(self.wide(wide))], dim=1
        )
        return linear + self.out_proj(context)


class DualBandFusion(nn.Module):
    """Direct L2 fusion; modes differ only in their low-frequency message."""

    def __init__(
        self,
        channels: int,
        mode: str,
        heads: int = 4,
        predictor_hidden: int = 32,
        source_values: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"difference", "prediction"}:
            raise ValueError(f"Unknown low-frequency message mode: {mode}")
        if source_values and mode != "prediction":
            raise ValueError("source_values is only supported in prediction mode")
        self.mode = mode
        self.source_values = source_values
        self.fdm_vis = LearnableFDM(channels, kernel_size=3, group=8)
        self.fdm_ir = LearnableFDM(channels, kernel_size=3, group=8)
        self.to_vis = ChannelCrossAttention(channels, heads=heads)
        self.to_ir = ChannelCrossAttention(channels, heads=heads)
        self.low_fuse = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1, bias=False), nn.GELU()
        )
        self.high_fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
            nn.GELU(),
        )
        self.out = nn.Conv2d(channels, channels, 1, bias=False)
        self._prediction_loss: Tensor | None = None

        if mode == "prediction":
            with torch.random.fork_rng(devices=[]):
                self.pred_v_to_i = SplitScalePredictor(channels, predictor_hidden)
                self.pred_i_to_v = SplitScalePredictor(channels, predictor_hidden)

    def forward(self, vis: Tensor, ir: Tensor) -> Tensor:
        self._prediction_loss = None
        vis_low, vis_high = self.fdm_vis(vis)
        ir_low, ir_high = self.fdm_ir(ir)

        if self.mode == "prediction":
            pred_ir = self.pred_v_to_i(vis_low.detach())
            pred_vis = self.pred_i_to_v(ir_low.detach())
            if self.training and torch.is_grad_enabled():
                self._prediction_loss = F.mse_loss(
                    pred_ir, ir_low.detach()
                ) + F.mse_loss(pred_vis, vis_low.detach())
            message_ir = ir_low - pred_ir.detach()
            message_vis = vis_low - pred_vis.detach()
        else:
            message_ir = ir_low - vis_low.detach()
            message_vis = vis_low - ir_low.detach()

        if self.source_values:
            context_vis = self.to_vis(vis_low, message_ir, ir_low)
            context_ir = self.to_ir(ir_low, message_vis, vis_low)
        else:
            context_vis = self.to_vis(vis_low, message_ir)
            context_ir = self.to_ir(ir_low, message_vis)
        low = self.low_fuse(
            torch.cat([vis_low, ir_low, context_vis, context_ir], dim=1)
        )
        high = self.high_fuse(torch.cat([vis_high, ir_high], dim=1))
        return self.out(low + high)

    def pop_prediction_loss(self) -> Tensor | None:
        loss = self._prediction_loss
        self._prediction_loss = None
        return loss


__all__ = [
    "LearnableFDM",
    "ChannelCrossAttention",
    "SplitScalePredictor",
    "DualBandFusion",
]
