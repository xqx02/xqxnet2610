import torch
import torch.nn.functional as F

from fusion.models.dual_band_fusion import (
    ChannelCrossAttention,
    DualBandFusion,
    LearnableFDM,
)


def _unfold_reference(module: LearnableFDM, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = x.shape
    kernel = module.norm(module.filter_gen(module.pool(x)))
    kernel = kernel.reshape(batch, module.group, 1, module.kernel_area, 1)
    kernel = torch.softmax(kernel, dim=3)
    patches = F.unfold(module.pad(x), kernel_size=module.kernel_size)
    patches = patches.reshape(
        batch,
        module.group,
        channels // module.group,
        module.kernel_area,
        height * width,
    )
    low = (patches * kernel).sum(dim=3).reshape(batch, channels, height, width)
    return low, x - low


def test_dynamic_fdm_matches_unfold_outputs_and_gradients() -> None:
    torch.manual_seed(7)
    module = LearnableFDM(16, group=4).double()
    with torch.no_grad():
        module.filter_gen.weight.normal_(0.0, 0.07)
        module.norm.weight.uniform_(0.6, 1.4)
        module.norm.bias.uniform_(-0.3, 0.3)

    reference_input = torch.randn(2, 16, 9, 13, dtype=torch.double, requires_grad=True)
    dynamic_input = reference_input.detach().clone().requires_grad_(True)
    reference = _unfold_reference(module, reference_input)
    dynamic = module(dynamic_input)
    probes = [torch.randn_like(value) for value in reference]
    reference_grads = torch.autograd.grad(
        sum((value * probe).sum() for value, probe in zip(reference, probes)),
        (reference_input, *module.parameters()),
        retain_graph=True,
    )
    dynamic_grads = torch.autograd.grad(
        sum((value * probe).sum() for value, probe in zip(dynamic, probes)),
        (dynamic_input, *module.parameters()),
    )

    for expected, actual in zip(reference, dynamic):
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
    for expected, actual in zip(reference_grads, dynamic_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(dynamic[0] + dynamic[1], dynamic_input)


def test_f1_f2_share_initial_output_and_isolate_prediction_gradients() -> None:
    torch.manual_seed(2026)
    f1 = DualBandFusion(16, mode="difference", heads=4, predictor_hidden=8)
    rng_after_f1 = torch.get_rng_state()
    torch.manual_seed(2026)
    f2 = DualBandFusion(16, mode="prediction", heads=4, predictor_hidden=8)
    rng_after_f2 = torch.get_rng_state()
    assert torch.equal(rng_after_f1, rng_after_f2)
    for key, value in f1.state_dict().items():
        torch.testing.assert_close(value, f2.state_dict()[key], rtol=0, atol=0)

    vis = torch.randn(2, 16, 8, 10, requires_grad=True)
    ir = torch.randn(2, 16, 8, 10, requires_grad=True)
    f1_output = f1(vis, ir)
    f2_output = f2(vis, ir)
    torch.testing.assert_close(f1_output, f2_output, rtol=0, atol=0)

    prediction_loss = f2.pop_prediction_loss()
    assert prediction_loss is not None
    prediction_loss.backward()
    predictor_prefixes = ("pred_v_to_i.", "pred_i_to_v.")
    assert all(
        parameter.grad is None
        for name, parameter in f2.named_parameters()
        if not name.startswith(predictor_prefixes)
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in f2.named_parameters()
        if name.startswith(predictor_prefixes)
    )


def test_split_value_attention_matches_legacy_when_sources_match() -> None:
    torch.manual_seed(19)
    module = ChannelCrossAttention(16, heads=4).double()
    query_legacy = torch.randn(2, 16, 7, 9, dtype=torch.double, requires_grad=True)
    message_legacy = torch.randn(2, 16, 7, 9, dtype=torch.double, requires_grad=True)
    query_split = query_legacy.detach().clone().requires_grad_(True)
    message_split = message_legacy.detach().clone().requires_grad_(True)

    legacy = module(query_legacy, message_legacy)
    split = module(query_split, message_split, message_split)
    probe = torch.randn_like(legacy)
    parameters = tuple(module.parameters())
    legacy_grads = torch.autograd.grad(
        (legacy * probe).sum(), (query_legacy, message_legacy, *parameters), retain_graph=True
    )
    split_grads = torch.autograd.grad(
        (split * probe).sum(), (query_split, message_split, *parameters)
    )

    torch.testing.assert_close(split, legacy, rtol=1e-12, atol=1e-12)
    for expected, actual in zip(legacy_grads, split_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_source_values_preserve_parameters_and_gradient_routes() -> None:
    torch.manual_seed(2026)
    f2 = DualBandFusion(64, mode="prediction")
    rng_after_f2 = torch.get_rng_state()
    torch.manual_seed(2026)
    f3 = DualBandFusion(64, mode="prediction", source_values=True)
    rng_after_f3 = torch.get_rng_state()

    assert torch.equal(rng_after_f2, rng_after_f3)
    assert sum(parameter.numel() for parameter in f2.parameters()) == 83_752
    assert sum(parameter.numel() for parameter in f3.parameters()) == 83_752
    assert sum(parameter.numel() for parameter in f3.pred_v_to_i.parameters()) == 8_480
    for key, value in f2.state_dict().items():
        torch.testing.assert_close(value, f3.state_dict()[key], rtol=0, atol=0)

    attention = ChannelCrossAttention(16, heads=4)
    query = torch.randn(2, 16, 8, 10, requires_grad=True)
    message = torch.randn(2, 16, 8, 10, requires_grad=True)
    value = torch.randn(2, 16, 8, 10, requires_grad=True)
    attention(query, message, value).square().mean().backward()
    assert value.grad is not None and value.grad.abs().sum() > 0
    assert attention.kv.weight.grad is not None
    assert attention.kv.weight.grad[:16].abs().sum() > 0
    assert attention.kv.weight.grad[16:].abs().sum() > 0

    fusion = DualBandFusion(
        16,
        mode="prediction",
        heads=4,
        predictor_hidden=8,
        source_values=True,
    )
    vis = torch.randn(2, 16, 8, 10, requires_grad=True)
    ir = torch.randn(2, 16, 8, 10, requires_grad=True)
    fusion(vis, ir).square().mean().backward()
    predictor_prefixes = ("pred_v_to_i.", "pred_i_to_v.")
    assert all(
        parameter.grad is None
        for name, parameter in fusion.named_parameters()
        if name.startswith(predictor_prefixes)
    )
    assert vis.grad is not None and vis.grad.abs().sum() > 0
    assert ir.grad is not None and ir.grad.abs().sum() > 0
    assert fusion.pop_prediction_loss() is not None
    assert fusion.pop_prediction_loss() is None
