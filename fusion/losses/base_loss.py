from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F


def _to_gray(x: Tensor) -> Tensor:
    if x.shape[1] == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _grad_components(x: Tensor) -> tuple[Tensor, Tensor]:
    kernel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    x = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x, kernel_x), F.conv2d(x, kernel_y)


__all__ = ["_to_gray", "_grad_components"]
