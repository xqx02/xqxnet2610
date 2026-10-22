from __future__ import annotations

from pathlib import Path

import torch

from fusion.dataset import _stratified_split_names
from fusion.models.plain_fusion import ConcatFuse, PlainFusionNet


def test_concat_fuse_shape_and_grad() -> None:
    fuse = ConcatFuse(8)
    a = torch.randn(2, 8, 17, 19, requires_grad=True)
    b = torch.randn(2, 8, 17, 19, requires_grad=True)
    y = fuse(a, b)
    assert y.shape == a.shape
    y.mean().backward()
    assert a.grad is not None
    assert b.grad is not None


def test_plain_fusion_has_no_innovation_modules() -> None:
    model = PlainFusionNet(base_channels=8, encoder_blocks=(1, 1, 1, 1))
    assert not hasattr(model, "csp1")
    assert not hasattr(model, "rfgm2")
    vis = torch.rand(1, 3, 31, 29)
    ir = torch.rand(1, 1, 31, 29)
    out = model(vis, ir)
    assert out.shape == (1, 1, 31, 29)
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_day_night_split_is_fixed_and_stratified() -> None:
    names = [f"{index:05d}D.pt" for index in range(536)] + [f"{index:05d}N.pt" for index in range(547)]
    train, val = _stratified_split_names(names, val_ratio=0.1, seed=2026)
    train_again, val_again = _stratified_split_names(names, val_ratio=0.1, seed=2026)
    assert (train, val) == (train_again, val_again)
    assert len(train) == 975 and len(val) == 108
    assert sum(Path(name).stem.endswith("D") for name in val) == 53
    assert sum(Path(name).stem.endswith("N") for name in val) == 55
    assert set(train).isdisjoint(val)


