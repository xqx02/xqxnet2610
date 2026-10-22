from __future__ import annotations

import torch

from fusion.losses import LOSS_REGISTRY, build_loss
from fusion.losses.loss_choice import _controlfusion_ssim_loss


def test_only_c0_is_registered_and_backward_is_finite() -> None:
    assert set(LOSS_REGISTRY) == {"controlfusion_c0_yadapter"}
    loss = build_loss(
        "controlfusion_c0_yadapter",
        intensity_weight=8.0,
        gradient_weight=10.0,
        ssim_weight=1.0,
        color_weight=12.0,
    )
    fused = torch.rand(1, 1, 48, 48, requires_grad=True)
    vis = torch.rand(1, 3, 48, 48)
    ir = torch.rand(1, 1, 48, 48)
    output = loss(fused, vis, ir)
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert fused.grad is not None and torch.isfinite(fused.grad).all()


def test_c0_ssim_identity_is_zero() -> None:
    image = torch.rand(1, 3, 48, 48)
    assert torch.allclose(_controlfusion_ssim_loss(image, image), torch.zeros(()), atol=1e-6)
