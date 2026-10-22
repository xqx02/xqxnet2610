from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fusion.config import load_config
from fusion.dataset import (
    OriginalFusionValidationDataset,
    PairedFusionDataset,
    TensorFusionDataset,
    fixed_split_tensor_names,
    split_tensor_names,
)
from fusion.engine import evaluate_loss, output_gradient_diagnostics, train_one_epoch
from fusion.losses import build_loss
from fusion.models import build_model
from fusion.validation_metrics import CORE_METRICS, evaluate_fusion_metrics


def _last_conv_weight(model: torch.nn.Module) -> torch.nn.Parameter:
    head = getattr(getattr(model, "decoder", None), "head", None)
    weights = [module.weight for module in head.modules() if isinstance(module, torch.nn.Conv2d)] if head else []
    if not weights:
        raise RuntimeError("Training health check requires a Conv2d output head.")
    return weights[-1]


def _format_key_metrics(metrics: dict[str, float], prefix: str) -> str:
    selected = {
        "optimizer_steps",
        "nonzero_grad_steps",
        "grad_norm_mean",
        "output_lower_saturation_mean",
        "output_upper_saturation_mean",
        "intensity",
        "gradient",
        "ssim",
        "weighted_intensity",
        "weighted_gradient",
        "weighted_ssim",
    }
    selected.update(key for key in metrics if key.endswith("_high_contribution_mean"))
    selected.update(key for key in metrics if key.endswith("_low_weight_mean"))
    selected.update(key for key in metrics if key.startswith("csp") and key.endswith("_mean"))
    selected.update(key for key in metrics if key.startswith("rfgm") and key.endswith("_mean"))
    message = ""
    for key in sorted(selected & metrics.keys()):
        value = metrics[key]
        precision = ".3e" if "ratio" in key or "norm" in key or "grad" in key else ".4f"
        message += f" {prefix}_{key}={value:{precision}}"
    return message


def check_training_health(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> None:
    """Fail early if hard output clipping has disconnected the training graph."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.train()
    model.zero_grad(set_to_none=True)
    vis, ir = next(iter(loader))
    vis = vis.to(device, non_blocking=True)
    ir = ir.to(device, non_blocking=True)
    fused = model(vis, ir)
    fusion_loss = criterion(fused, vis, ir)["loss"]
    pop_prediction_loss = getattr(model, "pop_prediction_loss", None)
    prediction_loss = pop_prediction_loss() if callable(pop_prediction_loss) else None
    loss = fusion_loss if prediction_loss is None else fusion_loss + prediction_loss
    loss.backward()
    head_grad = _last_conv_weight(model).grad
    grad_norm = 0.0 if head_grad is None else float(head_grad.detach().float().norm().cpu())
    lower = float((fused <= 0.0).float().mean().cpu())
    upper = float((fused >= 1.0).float().mean().cpu())
    predictor_grad_norm = 0.0
    if prediction_loss is not None:
        predictor_gradients = [
            parameter.grad.detach().float().norm()
            for name, parameter in model.named_parameters()
            if name.startswith(("fuse2.pred_v_to_i.", "fuse2.pred_i_to_v."))
            and parameter.grad is not None
        ]
        if predictor_gradients:
            predictor_grad_norm = float(
                torch.linalg.vector_norm(torch.stack(predictor_gradients)).cpu()
            )
    print(
        f"preflight output_min={float(fused.min()):.6f} output_max={float(fused.max()):.6f} "
        f"lower_clamp_ratio={lower:.4f} upper_clamp_ratio={upper:.4f} "
        f"head_grad_norm={grad_norm:.3e} predictor_grad_norm={predictor_grad_norm:.3e}"
    )
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if grad_norm == 0.0:
        raise RuntimeError("Preflight failed: output head gradient is zero. Training would not update parameters.")
    if prediction_loss is not None and predictor_grad_norm == 0.0:
        raise RuntimeError("Preflight failed: prediction loss does not update the predictors.")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    epoch: int,
    best_val_loss: float,
    cfg: dict,
    metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": cfg,
            "train_metrics": metrics,
            "val_metrics": val_metrics,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def _descending_ranks(values: list[float]) -> list[float]:
    """Return 1-based descending ranks, assigning tied values their mean rank."""
    ranks = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        mean_rank = (start + 1 + end) / 2.0
        for index, _ in ordered[start:end]:
            ranks[index] = mean_rank
        start = end
    return ranks


def _split_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def initialize_backbone(model: torch.nn.Module, cfg: dict, path: Path, report_path: Path) -> None:
    """Load the exact P0 backbone and restore P0's post-construction CPU RNG."""
    initial_file = torch.load(path, map_location="cpu")
    initial = initial_file["model"] if isinstance(initial_file, dict) and "model" in initial_file else initial_file
    initial = {key.replace("module.", ""): value for key, value in initial.items()}
    kwargs = {
        "base_channels": int(cfg["base_channels"]),
        "encoder_blocks": tuple(cfg["encoder_blocks"]),
        "attention_heads": tuple(cfg.get("attention_heads", [1, 2, 4, 8])),
        "ffn_expansion": float(cfg.get("ffn_expansion", 2.66)),
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(cfg["train"]["seed"]))
        reference = build_model("plain_fusion", **kwargs)
        rng_after_plain = torch.get_rng_state().clone()
    reference_state = reference.state_dict()
    if set(initial) != set(reference_state) or any(
        initial[key].shape != reference_state[key].shape or not torch.equal(initial[key], reference_state[key])
        for key in initial
    ):
        raise RuntimeError("Shared initialization does not match the seeded P0 backbone.")

    state = model.state_dict()
    if not set(initial).issubset(state):
        raise RuntimeError("Experiment model is missing P0 backbone parameters.")
    extra = sorted(set(state) - set(initial))
    extra_prefixes = sorted({key.split(".", 1)[0] for key in extra})
    expected_prefixes = {
        "plain_conditional_csp_l1_fusion": {"csp1"},
        "plain_original_rfgm_l2_fusion": {"rfgm2"},
    }.get(cfg["model"])
    if not extra or set(extra_prefixes) != expected_prefixes:
        raise RuntimeError(
            f"Unexpected experiment modules for {cfg['model']}: "
            f"found={extra_prefixes}, expected={sorted(expected_prefixes or [])}"
        )
    for key, value in initial.items():
        if state[key].shape != value.shape:
            raise RuntimeError(f"Backbone shape mismatch: {key}")
        state[key] = value.to(device=state[key].device, dtype=state[key].dtype)
    model.load_state_dict(state, strict=True)
    torch.set_rng_state(rng_after_plain)

    report = {
        "initial_file": str(path.resolve()),
        "initial_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": int(cfg["train"]["seed"]),
        "backbone_keys": len(initial),
        "cpu_rng_sha256": hashlib.sha256(rng_after_plain.numpy().tobytes()).hexdigest(),
    }
    if len(extra_prefixes) == 1:
        report["extra_prefix"] = extra_prefixes[0]
    else:
        report["extra_prefixes"] = extra_prefixes
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists() and json.loads(report_path.read_text(encoding="utf-8")) != report:
        raise RuntimeError(f"Backbone initialization report differs: {report_path}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def initialize_e1_base(model: torch.nn.Module, cfg: dict, path: Path, report_path: Path) -> None:
    """Load retained E1, with an exact fuse2 replacement rule for F1/F2/F3."""
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("E1 initialization requires a checkpoint saved by train.py.")
    source_cfg = checkpoint.get("config", {})
    if source_cfg.get("model") != "plain_conditional_csp_l1_fusion":
        raise RuntimeError("E1 initialization source is not the retained E1 model.")
    for key in ("base_channels", "encoder_blocks", "attention_heads", "ffn_expansion"):
        if source_cfg.get(key) != cfg.get(key):
            raise RuntimeError(f"E1 initialization architecture differs at {key}.")

    source = {key.replace("module.", ""): value for key, value in checkpoint["model"].items()}
    target = model.state_dict()
    replacement_models = {
        "plain_e1_dualband_diff_l2",
        "plain_e1_dualband_pred_l2",
        "plain_e1_dualband_pred_sourcev_l2",
    }
    if cfg["model"] in replacement_models:
        kwargs = {
            "base_channels": int(cfg["base_channels"]),
            "encoder_blocks": tuple(cfg["encoder_blocks"]),
            "attention_heads": tuple(cfg.get("attention_heads", [1, 2, 4, 8])),
            "ffn_expansion": float(cfg.get("ffn_expansion", 2.66)),
        }
        with torch.random.fork_rng(devices=[]):
            reference = build_model("plain_conditional_csp_l1_fusion", **kwargs)
        reference_state = reference.state_dict()
        if set(source) != set(reference_state) or any(
            source[key].shape != reference_state[key].shape for key in source
        ):
            raise RuntimeError("E1 initialization source is incomplete or structurally incompatible.")

        excluded = {
            "fuse2.op.0.weight",
            "fuse2.op.0.bias",
            "fuse2.op.2.weight",
            "fuse2.op.2.bias",
        }
        found = {key for key in source if key.startswith("fuse2.op.")}
        if found != excluded:
            raise RuntimeError(f"Unexpected retained E1 fuse2 parameters: {sorted(found)}")
        shared = set(source) - excluded
        if not shared.issubset(target):
            missing = sorted(shared - set(target))
            raise RuntimeError(f"F1/F2/F3 does not preserve the complete E1 base: {missing[:5]}")
        if any(key in target for key in excluded):
            raise RuntimeError("Old E1 fuse2 parameters remain in the replacement model.")
        added = sorted(set(target) - shared)
        if not added or any(not key.startswith("fuse2.") for key in added):
            raise RuntimeError(f"Unexpected F1/F2/F3 parameters outside fuse2: {added[:5]}")
        for key in shared:
            value = source[key]
            if target[key].shape != value.shape:
                raise RuntimeError(f"E1 base shape mismatch: {key}")
            target[key] = value.to(device=target[key].device, dtype=target[key].dtype)
        model.load_state_dict(target, strict=True)

        common_new = [
            key for key in added
            if not key.startswith(("fuse2.pred_v_to_i.", "fuse2.pred_i_to_v."))
        ]
        digest = hashlib.sha256()
        for key in common_new:
            value = target[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        report = {
            "base_model": "plain_conditional_csp_l1_fusion",
            "source_file": str(path.resolve()),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_epoch": int(checkpoint.get("epoch", 0)),
            "loaded_keys": len(shared),
            "excluded_keys": sorted(excluded),
            "new_keys": added,
            "new_fuse2_common_sha256": digest.hexdigest(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists() and json.loads(report_path.read_text(encoding="utf-8")) != report:
            raise RuntimeError(f"E1 initialization report differs: {report_path}")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    if not set(source).issubset(target):
        missing = sorted(set(source) - set(target))
        raise RuntimeError(f"Future experiment does not preserve the E1 base: {missing[:5]}")
    extra_prefixes = sorted({key.split(".", 1)[0] for key in set(target) - set(source)})
    if not extra_prefixes:
        raise RuntimeError("--init-e1 is reserved for models that extend E1 with new modules.")
    for key, value in source.items():
        if target[key].shape != value.shape:
            raise RuntimeError(f"E1 base shape mismatch: {key}")
        target[key] = value.to(device=target[key].device, dtype=target[key].dtype)
    model.load_state_dict(target, strict=True)

    report = {
        "base_model": "plain_conditional_csp_l1_fusion",
        "source_file": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_epoch": int(checkpoint.get("epoch", 0)),
        "base_keys": len(source),
        "extra_prefixes": extra_prefixes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists() and json.loads(report_path.read_text(encoding="utf-8")) != report:
        raise RuntimeError(f"E1 initialization report differs: {report_path}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _fixed_diagnostic_names(names: list[str], size: int, seed: int) -> list[str]:
    """Deterministically choose a day/night-proportional subset from training only."""
    if size <= 0:
        return []
    size = min(size, len(names))
    groups: dict[str, list[str]] = {}
    for name in names:
        suffix = Path(name).stem[-1:].upper()
        groups.setdefault(suffix if suffix in {"D", "N"} else "other", []).append(name)
    exact = {key: len(values) * size / len(names) for key, values in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    for key in sorted(groups, key=lambda item: (exact[item] - quotas[item], item), reverse=True)[: size - sum(quotas.values())]:
        quotas[key] += 1
    selected: list[str] = []
    for key, values in groups.items():
        ordered = sorted(
            values,
            key=lambda name: hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(),
        )
        selected.extend(ordered[: quotas[key]])
    return sorted(selected)


def _write_per_image_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("No per-image validation metrics to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_metric_checkpoint(
    candidate_path: Path,
    weights_dir: Path,
    selection_weights: dict[str, float],
    expected_epochs: list[int] | None = None,
    split_hash: str | None = None,
    validation_protocol: str | None = None,
) -> dict:
    """Rank all candidate epochs together; this is post-training model selection, not a loss."""
    rows = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No validation metric candidates found: {candidate_path}")
    if set(selection_weights) != set(CORE_METRICS):
        raise ValueError(f"selection_weights must contain exactly {CORE_METRICS}.")
    weights = {key: float(selection_weights[key]) for key in CORE_METRICS}
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()) or not math.isfinite(sum(weights.values())) or sum(weights.values()) <= 0.0:
        raise ValueError(f"selection_weights must be non-negative with positive sum: {weights}")
    epochs = [row.get("epoch") for row in rows]
    if any(type(epoch) is not int for epoch in epochs) or len(epochs) != len(set(epochs)):
        raise ValueError("Candidate epochs must be unique integers.")
    if expected_epochs is not None and sorted(epochs) != sorted(expected_epochs):
        raise ValueError(f"Candidate epochs differ from expected set: got={sorted(epochs)}, expected={sorted(expected_epochs)}")
    for row in rows:
        missing = [key for key in CORE_METRICS if key not in row]
        if missing:
            raise RuntimeError(f"Candidate epoch {row.get('epoch')} is missing metrics: {missing}")
        for key in CORE_METRICS:
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Candidate epoch {row['epoch']} has non-finite {key}.")
        if split_hash is not None and row.get("split_hash") != split_hash:
            raise ValueError(f"Candidate epoch {row['epoch']} has a different split hash.")
        if validation_protocol is not None and row.get("validation_protocol") != validation_protocol:
            raise ValueError(f"Candidate epoch {row['epoch']} has a different validation protocol.")
        checkpoint = weights_dir / f"epoch_{row['epoch']:03d}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Candidate checkpoint missing: {checkpoint}")
    for key in CORE_METRICS:
        for row, rank in zip(rows, _descending_ranks([float(item[key]) for item in rows])):
            row[f"rank_{key}"] = rank
    for row in rows:
        row["selection_score"] = sum(weights[key] * float(row[f"rank_{key}"]) for key in CORE_METRICS)
    selected = min(rows, key=lambda row: (row["selection_score"], -float(row["VIF"]), -float(row["QABF"]), -float(row["SSIM"]), int(row["epoch"])))
    checkpoint = weights_dir / f"epoch_{int(selected['epoch']):03d}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected metric checkpoint is missing: {checkpoint}")
    shutil.copy2(checkpoint, weights_dir / "best_metrics.pth")
    report = {
        "selection_rule": "weighted descending ranks; lower selection_score is better",
        "selection_weights": weights,
        "tie_break": ["higher VIF", "higher QABF", "higher SSIM", "earlier epoch"],
        "selected": selected,
        "gap_to_best": {
            key: max(float(row[key]) for row in rows) - float(selected[key])
            for key in CORE_METRICS
        },
        "split_hash": split_hash,
        "validation_protocol": validation_protocol,
        "candidates": rows,
        "checkpoint": str(checkpoint),
    }
    (weights_dir.parent / "validation_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (weights_dir.parent / "validation_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = max(warmup_epochs * steps_per_epoch, 0)
    if total_steps <= 1 and warmup_steps == 0:
        return None
    min_ratio = min_lr / lr if lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--tensor-dir", default=None)
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--ir-dir", default=None)
    parser.add_argument("--out", default="runs/fusion_clean.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="", help="Resume from a checkpoint saved by this script.")
    parser.add_argument("--init-weights", default="", help="Load a shared model state before creating the optimizer.")
    parser.add_argument("--init-backbone", default="", help="Load the shared P0 backbone into a module experiment.")
    parser.add_argument("--init-e1", default="", help="Load retained E1 into a future model that extends E1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initializers = (args.resume, args.init_weights, args.init_backbone, args.init_e1)
    if sum(bool(value) for value in initializers) > 1:
        raise ValueError("Use only one of --resume, --init-weights, --init-backbone, and --init-e1.")
    cfg = load_config(args.config)
    out = Path(args.out)
    torch.manual_seed(int(cfg.get("train", {}).get("seed", 3407)))
    model = build_model(
        cfg["model"],
        base_channels=int(cfg["base_channels"]),
        encoder_blocks=tuple(cfg["encoder_blocks"]),
        attention_heads=tuple(cfg.get("attention_heads", [1, 2, 4, 8])),
        ffn_expansion=float(cfg.get("ffn_expansion", 2.66)),
    )
    if args.init_weights:
        initial = torch.load(args.init_weights, map_location=args.device)
        state = initial["model"] if isinstance(initial, dict) and "model" in initial else initial
        model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()}, strict=True)
        print(f"Loaded shared initial weights from {args.init_weights}")
    elif args.init_backbone:
        initialize_backbone(model, cfg, Path(args.init_backbone), out.parent / "backbone_init.json")
        print(f"Loaded shared P0 backbone from {args.init_backbone}")
    elif args.init_e1:
        initialize_e1_base(model, cfg, Path(args.init_e1), out.parent / "e1_base_init.json")
        print(f"Loaded retained E1 base from {args.init_e1}")
    model = model.to(args.device)
    criterion = build_loss(cfg["loss"], **cfg.get("loss_kwargs", {})).to(args.device)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    diagnostic_cfg = cfg.get("diagnostics", {})
    tensor_dir = args.tensor_dir or data_cfg.get("tensor_dir")
    vis_dir = args.vis_dir or data_cfg.get("vis_dir")
    ir_dir = args.ir_dir or data_cfg.get("ir_dir")
    root = Path(__file__).resolve().parent
    if tensor_dir and not Path(tensor_dir).is_absolute():
        tensor_dir = str((root / tensor_dir).resolve())
    if vis_dir and not Path(vis_dir).is_absolute():
        vis_dir = str((root / vis_dir).resolve())
    if ir_dir and not Path(ir_dir).is_absolute():
        ir_dir = str((root / ir_dir).resolve())
    split_manifest: dict | None = None
    if tensor_dir:
        val_ratio = float(train_cfg.get("val_ratio", 0.1))
        split_file = data_cfg.get("split_file")
        if split_file:
            if val_ratio <= 0.0:
                raise ValueError("data.split_file requires train.val_ratio > 0 for development experiments.")
            split_path = Path(split_file)
            if not split_path.is_absolute():
                split_path = Path(__file__).resolve().parent / split_path
            train_names, val_names, split_manifest = fixed_split_tensor_names(tensor_dir, split_path)
            split_manifest["tensor_dir"] = str(Path(tensor_dir).resolve())
        else:
            split_seed = int(train_cfg.get("split_seed", train_cfg.get("seed", 3407)))
            train_names, val_names = split_tensor_names(tensor_dir, val_ratio, split_seed)
            count = lambda values, suffix: sum(Path(name).stem.upper().endswith(suffix) for name in values)
            split_manifest = {
                "tensor_dir": str(Path(tensor_dir).resolve()),
                "val_ratio": val_ratio,
                "split_seed": split_seed,
                "train_count": len(train_names),
                "val_count": len(val_names),
                "train_day": count(train_names, "D"),
                "train_night": count(train_names, "N"),
                "val_day": count(val_names, "D"),
                "val_night": count(val_names, "N"),
                "train_names": train_names,
                "val_names": val_names,
            }
        dataset = TensorFusionDataset(tensor_dir, train_names)
        val_dataset = TensorFusionDataset(tensor_dir, val_names) if val_names else None
        source_dir = data_cfg.get("raw_train_dir")
        if source_dir and not Path(source_dir).is_absolute():
            source_dir = str((root / source_dir).resolve())
        raw_val_dataset = OriginalFusionValidationDataset(source_dir, val_names) if source_dir and val_names else None
        diagnostic_size = int(diagnostic_cfg.get("train_subset_size", 0))
        diagnostic_names = _fixed_diagnostic_names(
            train_names,
            diagnostic_size,
            int(diagnostic_cfg.get("seed", train_cfg.get("seed", 2026))),
        )
        raw_train_diagnostic_dataset = (
            OriginalFusionValidationDataset(source_dir, diagnostic_names)
            if source_dir and diagnostic_names
            else None
        )
    elif vis_dir and ir_dir:
        dataset = PairedFusionDataset(vis_dir, ir_dir)
        val_dataset = None
        raw_val_dataset = None
        diagnostic_names = []
        raw_train_diagnostic_dataset = None
    else:
        raise ValueError("Provide either --tensor-dir or both --vis-dir/--ir-dir.")
    batch_size = int(train_cfg.get("batch_size", 4))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        if val_dataset is not None
        else None
    )
    raw_val_loader = (
        DataLoader(raw_val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        if raw_val_dataset is not None
        else None
    )
    raw_train_diagnostic_loader = (
        DataLoader(raw_train_diagnostic_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        if raw_train_diagnostic_dataset is not None
        else None
    )
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = int(train_cfg.get("epochs", 100))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    grad_clip_norm = grad_clip_norm if grad_clip_norm > 0 else None
    steps_per_epoch = max(math.ceil(len(loader) / max(grad_accum_steps, 1)), 1)
    scheduler = build_scheduler(
        optimizer,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=int(train_cfg.get("warmup_epochs", 0)),
        lr=lr,
        min_lr=float(train_cfg.get("min_lr", 0.0)),
    )
    weights_dir = out.parent / "weights"
    if split_manifest is not None:
        manifest_path = out.parent / "split.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != split_manifest:
                raise RuntimeError(f"Existing data split differs from current configuration: {manifest_path}")
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
        print(
            f"split train={split_manifest['train_count']} "
            f"(D={split_manifest['train_day']},N={split_manifest['train_night']}) "
            f"val={split_manifest['val_count']} "
            f"(D={split_manifest['val_day']},N={split_manifest['val_night']}) "
            f"manifest={manifest_path}"
        )
    if diagnostic_names:
        diagnostic_manifest = {
            "purpose": "training-only diagnostic; never used for checkpoint selection",
            "seed": int(diagnostic_cfg.get("seed", train_cfg.get("seed", 2026))),
            "count": len(diagnostic_names),
            "names": diagnostic_names,
            "split_hash": _split_hash(diagnostic_names),
        }
        diagnostic_manifest_path = out.parent / "diagnostic_train_names.json"
        if diagnostic_manifest_path.exists():
            existing = json.loads(diagnostic_manifest_path.read_text(encoding="utf-8"))
            if existing != diagnostic_manifest:
                raise RuntimeError(f"Existing diagnostic training subset differs: {diagnostic_manifest_path}")
        else:
            diagnostic_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_manifest_path.write_text(json.dumps(diagnostic_manifest, indent=2), encoding="utf-8")
    validation_cfg = cfg.get("validation", {})
    metric_every = int(validation_cfg.get("metric_every", 0))
    selection_weights = validation_cfg.get("selection_weights", {"SD": 0.1, "QABF": 0.3, "SSIM": 0.3, "VIF": 0.3})
    if metric_every > 0 and val_loader is None:
        raise ValueError("validation.metric_every requires a non-empty validation set.")
    if metric_every > 0 and raw_val_loader is None:
        raise ValueError("validation.metric_every requires data.raw_train_dir for original-size validation.")
    validation_protocol = str(validation_cfg.get("protocol", "msrs_original_full_resolution_ycrcb_uint8_v1"))
    split_hash = _split_hash(val_names) if tensor_dir else None
    diagnostic_every = int(diagnostic_cfg.get("every", 0))
    if diagnostic_every > 0 and raw_train_diagnostic_loader is None:
        raise ValueError("diagnostics.every requires diagnostics.train_subset_size > 0 and data.raw_train_dir.")
    save_every = int(train_cfg.get("save_every", 10))
    best_val_loss = float("inf")
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        if not isinstance(ckpt, dict) or ckpt.get("config") != cfg:
            raise RuntimeError("Resume checkpoint config does not match the requested experiment.")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()}, strict=True)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if isinstance(ckpt, dict) and scheduler is not None and ckpt.get("lr_scheduler") is not None:
            scheduler.load_state_dict(ckpt["lr_scheduler"])
        if isinstance(ckpt, dict):
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
            if ckpt.get("torch_rng_state") is not None:
                torch.set_rng_state(ckpt["torch_rng_state"].cpu())
            if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng_state"]])
        print(f"Resumed from {args.resume}; start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")

    metrics_path = out.parent / "metrics.jsonl"
    candidate_path = out.parent / "validation_candidates.jsonl"
    if start_epoch == 1:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")
        candidate_path.write_text("", encoding="utf-8")
    check_training_health(model, criterion, loader, args.device)

    metrics: dict[str, float] = {}
    val_metrics: dict[str, float] | None = None
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        metrics = train_one_epoch(
            model,
            criterion,
            loader,
            optimizer,
            args.device,
            grad_accum_steps=grad_accum_steps,
            grad_clip_norm=grad_clip_norm,
            scheduler=scheduler,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - epoch_start
        message = f"epoch={epoch:03d} lr={lr_now:.3e} train_loss={metrics['loss']:.6f}"
        message += _format_key_metrics(metrics, "train")
        fusion_metrics: dict[str, float] | None = None
        diagnostics: dict[str, object] | None = None
        is_candidate = val_loader is not None and metric_every > 0 and (epoch % metric_every == 0 or epoch == epochs)
        if val_loader is not None:
            val_metrics = evaluate_loss(model, criterion, val_loader, args.device)
            message += f" val_loss={val_metrics['loss']:.6f}"
            message += _format_key_metrics(val_metrics, "val")
            score = val_metrics["loss"]
            if is_candidate:
                fusion_metrics, per_image_metrics = evaluate_fusion_metrics(
                    model, raw_val_loader, val_names, args.device
                )
                _write_per_image_metrics(
                    out.parent / f"validation_metrics_epoch_{epoch:03d}.csv",
                    [
                        {
                            "epoch": epoch,
                            "split_hash": split_hash,
                            "validation_protocol": validation_protocol,
                            **row,
                        }
                        for row in per_image_metrics
                    ],
                )
                message += " val_metrics=" + ",".join(
                    f"{key}={fusion_metrics[key]:.6f}" for key in CORE_METRICS
                )
        else:
            val_metrics = None
            score = metrics["loss"]
        is_diagnostic = raw_train_diagnostic_loader is not None and diagnostic_every > 0 and (
            epoch % diagnostic_every == 0 or epoch == epochs
        )
        if is_diagnostic:
            train_image_metrics, train_image_rows = evaluate_fusion_metrics(
                model,
                raw_train_diagnostic_loader,
                diagnostic_names,
                args.device,
                compute_vif=False,
            )
            output_gradients = output_gradient_diagnostics(
                model,
                criterion,
                raw_train_diagnostic_loader,
                args.device,
            )
            _write_per_image_metrics(
                out.parent / f"training_diagnostics_epoch_{epoch:03d}.csv",
                [{"epoch": epoch, "subset_hash": _split_hash(diagnostic_names), **row} for row in train_image_rows],
            )
            diagnostics = {
                "train_image_metrics": train_image_metrics,
                "output_gradients": output_gradients,
            }
            message += (
                f" diag_SD={train_image_metrics['SD']:.6f}"
                f" diag_SSIM={train_image_metrics['SSIM']:.6f}"
            )
        message += f" time={epoch_seconds:.1f}s"
        print(message)
        record = {
            "epoch": epoch,
            "lr": lr_now,
            "seconds": epoch_seconds,
            "train": metrics,
            "val": val_metrics,
            "validation_fusion": fusion_metrics,
            "diagnostics": diagnostics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if score < best_val_loss:
            best_val_loss = score
            save_checkpoint(weights_dir / "best_loss.pth", model, optimizer, scheduler, epoch, best_val_loss, cfg, metrics, val_metrics)
            save_checkpoint(weights_dir / "best.pth", model, optimizer, scheduler, epoch, best_val_loss, cfg, metrics, val_metrics)
            print(f"Saved {weights_dir / 'best_loss.pth'}")
        latest_path = weights_dir / "latest.pth"
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_val_loss, cfg, metrics, val_metrics)
        if (save_every > 0 and (epoch % save_every == 0 or epoch == epochs)) or is_candidate:
            epoch_path = weights_dir / f"epoch_{epoch:03d}.pth"
            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                cfg,
                metrics,
                val_metrics,
            )
            print(f"Saved {epoch_path}")
        if fusion_metrics is not None:
            with candidate_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "split_hash": split_hash,
                    "validation_protocol": validation_protocol,
                    **fusion_metrics,
                }) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out, model, optimizer, scheduler, epochs, best_val_loss, cfg, metrics, val_metrics)
    if metric_every > 0:
        expected_epochs = list(range(metric_every, epochs + 1, metric_every))
        if epochs not in expected_epochs:
            expected_epochs.append(epochs)
        selected = select_metric_checkpoint(
            candidate_path,
            weights_dir,
            selection_weights,
            expected_epochs=expected_epochs,
            split_hash=split_hash,
            validation_protocol=validation_protocol,
        )
        print(
            f"Selected metrics checkpoint epoch={selected['selected']['epoch']} "
            f"score={selected['selected']['selection_score']:.4f} "
            f"path={weights_dir / 'best_metrics.pth'}"
        )
    print(f"Saved final checkpoint to {out}")


if __name__ == "__main__":
    main()
