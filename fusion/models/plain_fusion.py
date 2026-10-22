from __future__ import annotations

import torch
from torch import Tensor, nn

from .backbone import DecoderCore, Encoder, crop_to_size, pad_to_multiple


class ConcatFuse(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.op(torch.cat([a, b], dim=1))


class PlainFusionNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        encoder_blocks: tuple[int, int, int, int] = (2, 3, 3, 4),
        attention_heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion: float = 2.66,
    ) -> None:
        super().__init__()
        c = base_channels
        self.vis_encoder = Encoder(3, c, encoder_blocks, attention_heads, ffn_expansion)
        self.ir_encoder = Encoder(1, c, encoder_blocks, attention_heads, ffn_expansion)
        self.fuse1 = ConcatFuse(c)
        self.fuse2 = ConcatFuse(c * 2)
        self.fuse3 = ConcatFuse(c * 4)
        self.fuse4 = ConcatFuse(c * 8)
        self.decoder = DecoderCore(c, attention_heads, ffn_expansion)

    def prepare_l1(self, vis_l1: Tensor, ir_l1: Tensor) -> tuple[Tensor, Tensor]:
        return vis_l1, ir_l1

    def prepare_l2(self, vis_l2: Tensor, ir_l2: Tensor) -> tuple[Tensor, Tensor]:
        return vis_l2, ir_l2

    def refine_l2(self, current: Tensor, prior: Tensor) -> Tensor:
        return current

    def refine_l1(self, current: Tensor, prior: Tensor) -> Tensor:
        return current

    def forward(self, vis: Tensor, ir: Tensor) -> Tensor:
        vis, out_size = pad_to_multiple(vis, multiple=8)
        ir, _ = pad_to_multiple(ir, multiple=8)

        a1, a2, a3, a4 = self.vis_encoder(vis)
        b1, b2, b3, b4 = self.ir_encoder(ir)

        f4 = self.fuse4(a4, b4)
        x4 = self.decoder.dec4(f4)

        f3 = self.fuse3(a3, b3)
        x3 = self.decoder.up43(x4, f3.shape[-2:])
        x3 = self.decoder.dec3(self.decoder.merge3(x3, f3))

        a2, b2 = self.prepare_l2(a2, b2)
        f2 = self.fuse2(a2, b2)
        x2 = self.decoder.up32(x3, f2.shape[-2:])
        x2 = self.decoder.merge2(x2, f2)
        x2 = self.decoder.dec2(self.refine_l2(x2, f2))

        a1, b1 = self.prepare_l1(a1, b1)
        f1 = self.fuse1(a1, b1)
        x1 = self.decoder.up21(x2, f1.shape[-2:])
        x1 = self.decoder.merge1(x1, f1)
        x1 = self.decoder.dec1(self.refine_l1(x1, f1))
        x1 = self.decoder.refine(x1)

        return crop_to_size(self.decoder.head(x1), out_size)


__all__ = ["ConcatFuse", "PlainFusionNet"]
