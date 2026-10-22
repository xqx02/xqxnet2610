from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from .models.conditional_spectral_prior import ConditionalCrossModalSpectralPrior
from .models.encoder_fusion_rfgm import EncoderFusionRFGMRefine


_TRACKED_MODULES = (
    ConditionalCrossModalSpectralPrior,
    EncoderFusionRFGMRefine,
)


def _tracked_modules(model: nn.Module) -> dict[str, nn.Module]:
    return {name: module for name, module in model.named_modules() if isinstance(module, _TRACKED_MODULES)}


def _collect_forward_statistics(model: nn.Module) -> dict[str, torch.Tensor]:
    stats: dict[str, torch.Tensor] = {}
    for name, module in _tracked_modules(model).items():
        if isinstance(module, ConditionalCrossModalSpectralPrior) and module.last_statistics:
            for key, value in module.last_statistics.items():
                value = value.detach().float()
                # The conditional gates are Bx1xHxW maps.  Keeping every
                # pixel from every batch makes the epoch-end quantile both
                # unnecessarily expensive and, at L1 resolution, larger
                # than torch.quantile supports.  Reduce each image to its
                # spatial mean: the reported mean remains exact, while p95
                # becomes the 95th percentile across images, which is the
                # useful unit for a training diagnostic.
                if key.startswith(("spectral_gate", "local_gate")):
                    value = value.flatten(1).mean(dim=1)
                stats[f"{name}_{key}"] = value.flatten().cpu()
        elif isinstance(module, EncoderFusionRFGMRefine) and module.last_stats:
            for key, value in module.last_stats.items():
                stats[f"{name}_{key}"] = torch.as_tensor(value, dtype=torch.float32).reshape(1)
            stats[f"{name}_ratio"] = module.last_residual_ratio.detach().float().flatten().cpu()
    return stats


def _summary(chunks: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, values in chunks.items():
        if values:
            merged = torch.cat(values)
            metrics[f"{name}_mean"] = float(merged.mean())
            metrics[f"{name}_p95"] = float(torch.quantile(merged, 0.95))
    return metrics


def _module_grad_norm(module: nn.Module) -> float:
    norms = [parameter.grad.detach().float().norm() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms)).cpu()) if norms else 0.0


def _tracked_gradient_metrics(module: nn.Module) -> dict[str, float]:
    metrics = {"grad_norm": _module_grad_norm(module)}
    if isinstance(module, ConditionalCrossModalSpectralPrior):
        for direction, parameter in (("alpha_ir_grad", module.alpha_ir_theta), ("alpha_vis_grad", module.alpha_vis_theta)):
            metrics[direction] = 0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm().cpu())
    elif isinstance(module, EncoderFusionRFGMRefine):
        parameter = module.alpha_logit
        metrics["alpha_grad"] = 0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm().cpu())
        metrics["projection_grad_norm"] = _module_grad_norm(module.res_proj)
    return metrics


def _snapshot_modules(modules: dict[str, nn.Module]) -> dict[str, list[torch.Tensor]]:
    return {name: [parameter.detach().float().clone() for parameter in module.parameters()] for name, module in modules.items()}


def _parameter_change_metrics(modules: dict[str, nn.Module], snapshots: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, module in modules.items():
        before = snapshots[name]
        after = list(module.parameters())
        delta_sq = sum((current.detach().float() - initial).pow(2).sum() for current, initial in zip(after, before))
        base_sq = sum(initial.pow(2).sum() for initial in before)
        delta = float(delta_sq.sqrt())
        metrics[f"{name}_param_delta_norm"] = delta
        metrics[f"{name}_param_delta_ratio"] = delta / (float(base_sq.sqrt()) + 1e-12)
    return metrics


def _check_finite_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite gradient in {name}.")


def _total_grad_norm(model: nn.Module) -> float:
    norms = [parameter.grad.detach().float().norm() for parameter in model.parameters() if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms)).cpu()) if norms else 0.0


def _append_output_saturation(chunks: dict[str, list[torch.Tensor]], fused: torch.Tensor) -> None:
    chunks.setdefault("output_lower_saturation", []).append((fused <= 0.0).float().flatten(1).mean(dim=1).detach().cpu())
    chunks.setdefault("output_upper_saturation", []).append((fused >= 1.0).float().flatten(1).mean(dim=1).detach().cpu())


def _accumulate_loss_terms(sums: dict[str, float], losses: dict[str, torch.Tensor], batch_size: int) -> None:
    """Accumulate scalar loss terms by sample, never by number of batches."""
    for key, value in losses.items():
        if value.numel() != 1 or not torch.isfinite(value.detach()).all():
            raise FloatingPointError(f"Invalid loss component: {key}")
        sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    model.train()
    total = 0.0
    count = 0
    loss_sums: dict[str, float] = {}
    chunks: dict[str, list[torch.Tensor]] = {}
    modules = _tracked_modules(model)
    snapshots = _snapshot_modules(modules)
    grad_chunks: dict[str, list[torch.Tensor]] = {}
    calls = {name: 0 for name in modules}
    optimizer_steps = 0
    nonzero_grad_steps = 0
    grad_norm_total = 0.0
    grad_accum_steps = max(int(grad_accum_steps), 1)
    predictor_prefixes = ("fuse2.pred_v_to_i.", "fuse2.pred_i_to_v.")
    fusion_parameters: list[nn.Parameter] = []
    predictor_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameters = predictor_parameters if name.startswith(predictor_prefixes) else fusion_parameters
        parameters.append(parameter)
    optimizer.zero_grad(set_to_none=True)

    for step, (vis, ir) in enumerate(loader, start=1):
        vis = vis.to(device, non_blocking=True)
        ir = ir.to(device, non_blocking=True)
        fused = model(vis, ir)
        losses = criterion(fused, vis, ir)
        fusion_loss = losses["loss"]
        pop_prediction_loss = getattr(model, "pop_prediction_loss", None)
        prediction_loss = pop_prediction_loss() if callable(pop_prediction_loss) else None
        total_loss = fusion_loss if prediction_loss is None else fusion_loss + prediction_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite training loss.")
        (total_loss / grad_accum_steps).backward()
        _accumulate_loss_terms(loss_sums, losses, vis.size(0))
        if prediction_loss is not None:
            _accumulate_loss_terms(
                loss_sums,
                {"prediction_loss": prediction_loss, "total_loss": total_loss},
                vis.size(0),
            )
        _append_output_saturation(chunks, fused)
        for key, value in _collect_forward_statistics(model).items():
            chunks.setdefault(key, []).append(value)
        for name in calls:
            calls[name] += 1

        if step % grad_accum_steps == 0 or step == len(loader):
            _check_finite_gradients(model)
            for name, module in modules.items():
                for key, value in _tracked_gradient_metrics(module).items():
                    grad_chunks.setdefault(f"{name}_{key}", []).append(torch.tensor([value]))
            if grad_clip_norm is not None and grad_clip_norm > 0:
                fusion_norm = torch.nn.utils.clip_grad_norm_(fusion_parameters, grad_clip_norm)
                if predictor_parameters:
                    predictor_norm = torch.nn.utils.clip_grad_norm_(
                        predictor_parameters, grad_clip_norm
                    )
                    grad_norm = float(
                        torch.linalg.vector_norm(
                            torch.stack([fusion_norm.float(), predictor_norm.float()])
                        ).cpu()
                    )
                else:
                    grad_norm = float(fusion_norm.cpu())
            else:
                grad_norm = _total_grad_norm(model)
            optimizer.step()
            optimizer_steps += 1
            nonzero_grad_steps += int(grad_norm > 0.0)
            grad_norm_total += grad_norm
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        total += float(fusion_loss.detach().cpu()) * vis.size(0)
        count += vis.size(0)

    metrics = {
        "loss": total / max(count, 1),
        "optimizer_steps": float(optimizer_steps),
        "nonzero_grad_steps": float(nonzero_grad_steps),
        "grad_norm_mean": grad_norm_total / max(optimizer_steps, 1),
    }
    metrics.update({key: value / max(count, 1) for key, value in loss_sums.items()})
    metrics.update(_summary(chunks))
    metrics.update({f"{name}_calls": float(module_calls) for name, module_calls in calls.items()})
    for name, values in grad_chunks.items():
        metrics[f"{name}_mean"] = float(torch.cat(values).mean())
    metrics.update(_parameter_change_metrics(modules, snapshots))
    return metrics


@torch.no_grad()
def evaluate_loss(model: nn.Module, criterion: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total = 0.0
    count = 0
    loss_sums: dict[str, float] = {}
    chunks: dict[str, list[torch.Tensor]] = {}
    calls = {name: 0 for name in _tracked_modules(model)}
    for vis, ir in loader:
        vis = vis.to(device, non_blocking=True)
        ir = ir.to(device, non_blocking=True)
        fused = model(vis, ir)
        losses = criterion(fused, vis, ir)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Non-finite validation loss.")
        _accumulate_loss_terms(loss_sums, losses, vis.size(0))
        _append_output_saturation(chunks, fused)
        for key, value in _collect_forward_statistics(model).items():
            chunks.setdefault(key, []).append(value)
        for name in calls:
            calls[name] += 1
        total += float(losses["loss"].detach().cpu()) * vis.size(0)
        count += vis.size(0)
    metrics = {"loss": total / max(count, 1)}
    metrics.update({key: value / max(count, 1) for key, value in loss_sums.items()})
    metrics.update(_summary(chunks))
    metrics.update({f"{name}_calls": float(module_calls) for name, module_calls in calls.items()})
    return metrics


def output_gradient_diagnostics(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, float | None]:
    """Measure each weighted loss term's gradient with respect to fused output.

    This is a diagnostic only: model parameters are not connected to the
    gradient calculation and no optimizer state or parameter gradient changes.
    """
    required_terms = ("intensity", "gradient", "ssim")
    required_weights = ("intensity_weight", "gradient_weight", "ssim_weight")
    if not callable(getattr(criterion, "compute_terms", None)) or any(
        not hasattr(criterion, name) for name in required_weights
    ):
        raise TypeError("Gradient diagnostics require BaseFusionLoss-compatible criterion.")

    was_training = model.training
    model.eval()
    rms_sums = {key: 0.0 for key in required_terms}
    cosine_sums = {"intensity_gradient": 0.0, "intensity_ssim": 0.0, "gradient_ssim": 0.0}
    cosine_counts = {key: 0 for key in cosine_sums}
    count = 0
    weights = {
        "intensity": float(criterion.intensity_weight),
        "gradient": float(criterion.gradient_weight),
        "ssim": float(criterion.ssim_weight),
    }

    try:
        for vis, ir in loader:
            vis = vis.to(device, non_blocking=True)
            ir = ir.to(device, non_blocking=True)
            with torch.no_grad():
                fused = model(vis, ir)
            fused_leaf = fused.detach().float().requires_grad_(True)
            terms = criterion.compute_terms(fused_leaf, vis, ir)
            gradients: dict[str, torch.Tensor] = {}
            for key in required_terms:
                gradients[key] = torch.autograd.grad(
                    weights[key] * terms[key],
                    fused_leaf,
                    retain_graph=True,
                    create_graph=False,
                )[0].detach().float()

            batch_size = vis.size(0)
            for key, value in gradients.items():
                rms = value.flatten(1).square().mean(dim=1).sqrt()
                rms_sums[key] += float(rms.sum().cpu())
            for left, right in (("intensity", "gradient"), ("intensity", "ssim"), ("gradient", "ssim")):
                label = f"{left}_{right}"
                a = gradients[left].flatten(1)
                b = gradients[right].flatten(1)
                denominator = a.norm(dim=1) * b.norm(dim=1)
                valid = denominator > 0.0
                if valid.any():
                    cosine = (a[valid] * b[valid]).sum(dim=1) / denominator[valid]
                    cosine_sums[label] += float(cosine.sum().cpu())
                    cosine_counts[label] += int(valid.sum().cpu())
            count += batch_size
    finally:
        model.train(was_training)

    if count == 0:
        raise RuntimeError("Gradient diagnostic loader is empty.")
    result: dict[str, float | None] = {
        f"output_grad_rms_{key}": value / count for key, value in rms_sums.items()
    }
    total_rms = sum(value for value in result.values() if value is not None)
    for key in required_terms:
        result[f"output_grad_share_{key}"] = result[f"output_grad_rms_{key}"] / total_rms if total_rms > 0 else None
    for key, value in cosine_sums.items():
        result[f"output_grad_cos_{key}"] = value / cosine_counts[key] if cosine_counts[key] else None
        result[f"output_grad_cos_{key}_count"] = float(cosine_counts[key])
    return result
