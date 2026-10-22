from pathlib import Path

import torch

from fusion.models import MODEL_REGISTRY, build_model
from train import initialize_backbone, initialize_e1_base


def test_registry_contains_retained_and_dual_band_models() -> None:
    assert set(MODEL_REGISTRY) == {
        "plain_fusion",
        "plain_conditional_csp_l1_fusion",
        "plain_e1_dualband_diff_l2",
        "plain_e1_dualband_pred_l2",
        "plain_e1_dualband_pred_sourcev_l2",
        "plain_original_rfgm_l2_fusion",
    }


def test_retained_module_locations_and_forward() -> None:
    kwargs = dict(base_channels=8, encoder_blocks=(1, 1, 1, 1))
    p0 = build_model("plain_fusion", **kwargs).eval()
    e1 = build_model("plain_conditional_csp_l1_fusion", **kwargs).eval()
    e3 = build_model("plain_original_rfgm_l2_fusion", **kwargs).eval()
    f1 = build_model("plain_e1_dualband_diff_l2", **kwargs).eval()
    f2 = build_model("plain_e1_dualband_pred_l2", **kwargs).eval()
    f3 = build_model("plain_e1_dualband_pred_sourcev_l2", **kwargs).eval()
    assert not hasattr(p0, "csp1") and not hasattr(p0, "rfgm2")
    assert hasattr(e1, "csp1") and not hasattr(e1, "rfgm2")
    assert hasattr(e3, "rfgm2") and not hasattr(e3, "csp1")

    vis = torch.rand(1, 3, 32, 40)
    ir = torch.rand(1, 1, 32, 40)
    with torch.no_grad():
        assert p0(vis, ir).shape == (1, 1, 32, 40)
        assert e1(vis, ir).shape == (1, 1, 32, 40)
        assert e3(vis, ir).shape == (1, 1, 32, 40)
        assert f1(vis, ir).shape == (1, 1, 32, 40)
        assert f2(vis, ir).shape == (1, 1, 32, 40)
        assert f3(vis, ir).shape == (1, 1, 32, 40)
    assert e3.rfgm2.last_residual_ratio.item() > 0.0


def test_module_models_accept_the_shared_p0_backbone(tmp_path: Path) -> None:
    kwargs = dict(base_channels=8, encoder_blocks=(1, 1, 1, 1))
    torch.manual_seed(2026)
    initial = build_model("plain_fusion", **kwargs)
    initial_path = tmp_path / "initial.pth"
    torch.save(initial.state_dict(), initial_path)

    for model_name, prefix in (
        ("plain_conditional_csp_l1_fusion", "csp1"),
        ("plain_original_rfgm_l2_fusion", "rfgm2"),
    ):
        cfg = {
            "model": model_name,
            "base_channels": 8,
            "encoder_blocks": [1, 1, 1, 1],
            "attention_heads": [1, 2, 4, 8],
            "ffn_expansion": 2.66,
            "train": {"seed": 2026},
        }
        report_path = tmp_path / f"{prefix}.json"
        initialize_backbone(build_model(model_name, **kwargs), cfg, initial_path, report_path)
        assert f'"extra_prefix": "{prefix}"' in report_path.read_text(encoding="utf-8")


def test_future_model_can_initialize_from_complete_e1(tmp_path: Path) -> None:
    class FutureE1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = build_model(
                "plain_conditional_csp_l1_fusion",
                base_channels=8,
                encoder_blocks=(1, 1, 1, 1),
            )
            for name, module in self.base.named_children():
                setattr(self, name, module)
            del self.base
            self.future_module = torch.nn.Conv2d(8, 8, 1)

    architecture = {
        "base_channels": 8,
        "encoder_blocks": [1, 1, 1, 1],
        "attention_heads": [1, 2, 4, 8],
        "ffn_expansion": 2.66,
    }
    e1 = build_model(
        "plain_conditional_csp_l1_fusion",
        base_channels=8,
        encoder_blocks=(1, 1, 1, 1),
    )
    source = tmp_path / "e1.pth"
    torch.save({"model": e1.state_dict(), "config": {"model": "plain_conditional_csp_l1_fusion", **architecture}, "epoch": 100}, source)
    future = FutureE1()

    initialize_e1_base(future, {"model": "future_e1", **architecture}, source, tmp_path / "e1_base_init.json")

    for key, value in e1.state_dict().items():
        torch.testing.assert_close(future.state_dict()[key], value)


def test_f1_f2_replace_only_e1_fuse2(tmp_path: Path) -> None:
    architecture = {
        "base_channels": 8,
        "encoder_blocks": [1, 1, 1, 1],
        "attention_heads": [1, 2, 4, 8],
        "ffn_expansion": 2.66,
    }
    e1 = build_model(
        "plain_conditional_csp_l1_fusion",
        base_channels=8,
        encoder_blocks=(1, 1, 1, 1),
    )
    source = tmp_path / "e1.pth"
    torch.save(
        {
            "model": e1.state_dict(),
            "config": {"model": "plain_conditional_csp_l1_fusion", **architecture},
            "epoch": 100,
        },
        source,
    )

    for name in (
        "plain_e1_dualband_diff_l2",
        "plain_e1_dualband_pred_l2",
        "plain_e1_dualband_pred_sourcev_l2",
    ):
        model = build_model(name, base_channels=8, encoder_blocks=(1, 1, 1, 1))
        report = tmp_path / f"{name}.json"
        initialize_e1_base(model, {"model": name, **architecture}, source, report)
        for key, value in e1.state_dict().items():
            if not key.startswith("fuse2.op."):
                torch.testing.assert_close(model.state_dict()[key], value)
        assert not any(key.startswith("fuse2.op.") for key in model.state_dict())
        assert '"excluded_keys"' in report.read_text(encoding="utf-8")


def test_f2_f3_share_complete_e1_initialization(tmp_path: Path) -> None:
    architecture = {
        "base_channels": 8,
        "encoder_blocks": [1, 1, 1, 1],
        "attention_heads": [1, 2, 4, 8],
        "ffn_expansion": 2.66,
    }
    torch.manual_seed(11)
    e1 = build_model(
        "plain_conditional_csp_l1_fusion",
        base_channels=8,
        encoder_blocks=(1, 1, 1, 1),
    )
    source = tmp_path / "e1.pth"
    torch.save(
        {
            "model": e1.state_dict(),
            "config": {"model": "plain_conditional_csp_l1_fusion", **architecture},
            "epoch": 100,
        },
        source,
    )

    models = []
    rng_states = []
    for name in (
        "plain_e1_dualband_pred_l2",
        "plain_e1_dualband_pred_sourcev_l2",
    ):
        torch.manual_seed(2026)
        model = build_model(name, base_channels=8, encoder_blocks=(1, 1, 1, 1))
        initialize_e1_base(
            model,
            {"model": name, **architecture},
            source,
            tmp_path / f"{name}.json",
        )
        models.append(model)
        rng_states.append(torch.get_rng_state())

    assert torch.equal(rng_states[0], rng_states[1])
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)

    checkpoint = tmp_path / "f3.pth"
    torch.save(
        {
            "model": models[1].state_dict(),
            "config": {
                "model": "plain_e1_dualband_pred_sourcev_l2",
                **architecture,
            },
        },
        checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu")
    restored = build_model(
        saved["config"]["model"],
        base_channels=8,
        encoder_blocks=(1, 1, 1, 1),
    )
    restored.load_state_dict(saved["model"], strict=True)
