from __future__ import annotations

import torch
from torch import nn

from fusion.models.backbone import DecoderHead


class OldAdaptiveOutput(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dec_head = DecoderHead(channels).dec_head
        self.vis_base_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.ir_base_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.out_rho = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dec_raw = self.dec_head(x)
        base = 0.5 * (self.vis_base_head(x) + self.ir_base_head(x))
        return torch.clamp(self.out_rho * dec_raw + (1.0 - self.out_rho) * base, 0.0, 1.0)


def test_decoder_head_matches_old_rho_one_output() -> None:
    old = OldAdaptiveOutput(8)
    new = DecoderHead(8)
    new.dec_head.load_state_dict(old.dec_head.state_dict())
    x = torch.randn(2, 8, 11, 13)
    torch.testing.assert_close(new(x), old(x), atol=1e-6, rtol=1e-6)

