from __future__ import annotations

import torch
from torch import Tensor


def rgb_to_ycrcb(rgb: Tensor) -> Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5
    return torch.cat([y, cr, cb], dim=1)


def ycrcb_to_rgb(ycrcb: Tensor) -> Tensor:
    y, cr, cb = ycrcb[:, 0:1], ycrcb[:, 1:2] - 0.5, ycrcb[:, 2:3] - 0.5
    r = y + 1.403 * cr
    b = y + 1.773 * cb
    g = y - 0.714 * cr - 0.344 * cb
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)


def fuse_y_with_visible_chroma(fused_y: Tensor, vis_rgb: Tensor) -> Tensor:
    ycrcb = rgb_to_ycrcb(vis_rgb)
    return ycrcb_to_rgb(torch.cat([fused_y, ycrcb[:, 1:3]], dim=1))

