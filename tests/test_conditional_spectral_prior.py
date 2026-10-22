from __future__ import annotations

import torch

from fusion.models.conditional_spectral_prior import ConditionalCrossModalSpectralPrior
from fusion.models.module_screening import PlainConditionalCSPL1ScreeningNet


def test_conditional_csp_shape_grad_and_statistics() -> None:
    module = ConditionalCrossModalSpectralPrior(channels=8)
    vis = torch.randn(2, 8, 15, 17, requires_grad=True)
    ir = torch.randn(2, 8, 15, 17, requires_grad=True)

    out_vis, out_ir = module(vis, ir)
    assert out_vis.shape == vis.shape
    assert out_ir.shape == ir.shape
    assert module.last_statistics["band_gain_ir"].shape == (2, 4, 2)
    assert module.last_statistics["spectral_gate_ir"].shape == (2, 1, 15, 17)
    assert module.last_statistics["local_gate_ir"].shape == (2, 1, 15, 17)

    (out_vis.square().mean() + out_ir.square().mean()).backward()
    assert vis.grad is not None and torch.isfinite(vis.grad).all()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()
    assert module.alpha_ir_theta.grad is not None
    assert module.alpha_vis_theta.grad is not None
    assert module.vis_to_ir.band_context[-1].weight.grad is not None


def test_conditional_band_head_starts_from_static_half_gain() -> None:
    module = ConditionalCrossModalSpectralPrior(channels=8)
    vis = torch.randn(2, 8, 13, 16)
    ir = torch.randn(2, 8, 13, 16)
    module(vis, ir)
    torch.testing.assert_close(module.last_statistics["band_gain_ir"], torch.full((2, 4, 2), 0.5))
    torch.testing.assert_close(module.last_statistics["band_gain_vis"], torch.full((2, 4, 2), 0.5))


def test_conditional_csp_bypass_and_branch_modes() -> None:
    module = ConditionalCrossModalSpectralPrior(channels=8)
    vis = torch.randn(1, 8, 14, 18)
    ir = torch.randn(1, 8, 14, 18)

    module.diagnostic_mode = "bypass"
    out_vis, out_ir = module(vis, ir)
    torch.testing.assert_close(out_vis, vis, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_ir, ir, atol=0.0, rtol=0.0)

    module.diagnostic_mode = "spectral_only"
    spectral_vis, spectral_ir = module(vis, ir)
    module.diagnostic_mode = "local_only"
    local_vis, local_ir = module(vis, ir)
    assert float((spectral_vis - vis).abs().mean()) > 0.0
    assert float((spectral_ir - ir).abs().mean()) > 0.0
    assert float((local_vis - vis).abs().mean()) > 0.0
    assert float((local_ir - ir).abs().mean()) > 0.0


def test_e1_places_conditional_csp_only_at_l1() -> None:
    model = PlainConditionalCSPL1ScreeningNet(base_channels=8, encoder_blocks=(1, 1, 1, 1))
    assert isinstance(model.csp1, ConditionalCrossModalSpectralPrior)
    assert not hasattr(model, "csp2")
    assert not hasattr(model, "rfgm2")
    vis = torch.rand(1, 3, 31, 29)
    ir = torch.rand(1, 1, 31, 29)
    out = model(vis, ir)
    assert out.shape == (1, 1, 31, 29)
