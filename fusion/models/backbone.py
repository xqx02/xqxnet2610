from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def pad_to_multiple(x: Tensor, multiple: int = 8) -> tuple[Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (h, w)


def crop_to_size(x: Tensor, size: tuple[int, int]) -> Tensor:
    h, w = size
    return x[..., :h, :w]


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


def _num_heads(channels: int) -> int:
    for heads in (8, 4, 2):
        if channels % heads == 0:
            return heads
    return 1


class ChannelAttention(nn.Module):
    """Restormer-style transposed channel attention shared by baseline and main."""

    def __init__(self, channels: int, num_heads: int | None = None) -> None:
        super().__init__()
        self.num_heads = num_heads or _num_heads(channels)
        if channels % self.num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={self.num_heads}.")
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    """Shared backbone block; architectural comparisons change only fusion modules."""

    def __init__(self, channels: int, expansion: float = 2.66, num_heads: int | None = None) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.norm1 = LayerNorm2d(channels)
        self.attn = ChannelAttention(channels, num_heads=num_heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.proj(x)


class EncoderStage(nn.Module):
    def __init__(self, channels: int, depth: int, num_heads: int, expansion: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(channels, expansion=expansion, num_heads=num_heads) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        blocks: tuple[int, int, int, int] = (2, 3, 3, 4),
        attention_heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion: float = 2.66,
    ) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
        self.stage1 = EncoderStage(c, blocks[0], attention_heads[0], ffn_expansion)
        self.down12 = Downsample(c, c * 2)
        self.stage2 = EncoderStage(c * 2, blocks[1], attention_heads[1], ffn_expansion)
        self.down23 = Downsample(c * 2, c * 4)
        self.stage3 = EncoderStage(c * 4, blocks[2], attention_heads[2], ffn_expansion)
        self.down34 = Downsample(c * 4, c * 8)
        self.stage4 = EncoderStage(c * 8, blocks[3], attention_heads[3], ffn_expansion)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(self.down12(x1))
        x3 = self.stage3(self.down23(x2))
        x4 = self.stage4(self.down34(x3))
        return x1, x2, x3, x4


class FeatureMerge(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, decoder: Tensor, skip: Tensor) -> Tensor:
        return self.op(torch.cat([decoder, skip], dim=1))


class DecoderStage(nn.Module):
    def __init__(self, channels: int, depth: int = 1, num_heads: int | None = None, expansion: float = 2.66) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(channels, expansion=expansion, num_heads=num_heads) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class DecoderHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 1)
        self.dec_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        # Keep the initial prediction inside clamp's non-zero-gradient interval.
        # Inference still uses the original clamp(0, 1) output convention.
        nn.init.normal_(self.dec_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.dec_head[-1].bias, 0.5)

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(self.dec_head(x), 0.0, 1.0)


class DecoderCore(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        attention_heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion: float = 2.66,
    ) -> None:
        super().__init__()
        c = base_channels
        self.dec4 = DecoderStage(c * 8, depth=1, num_heads=attention_heads[3], expansion=ffn_expansion)
        self.up43 = Upsample(c * 8, c * 4)
        self.merge3 = FeatureMerge(c * 4)
        self.dec3 = DecoderStage(c * 4, depth=1, num_heads=attention_heads[2], expansion=ffn_expansion)
        self.up32 = Upsample(c * 4, c * 2)
        self.merge2 = FeatureMerge(c * 2)
        self.dec2 = DecoderStage(c * 2, depth=1, num_heads=attention_heads[1], expansion=ffn_expansion)
        self.up21 = Upsample(c * 2, c)
        self.merge1 = FeatureMerge(c)
        self.dec1 = DecoderStage(c, depth=1, num_heads=attention_heads[0], expansion=ffn_expansion)
        self.refine = TransformerBlock(c, expansion=ffn_expansion, num_heads=attention_heads[0])
        self.head = DecoderHead(c)
