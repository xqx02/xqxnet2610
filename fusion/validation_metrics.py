from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


CORE_METRICS = ("SD", "QABF", "SSIM", "VIF")


@lru_cache(maxsize=1)
def _evaluator():
    path = Path(__file__).resolve().parents[2] / "Fusion" / "tools" / "eval_msrs_smfnet_style.py"
    if not path.is_file():
        raise FileNotFoundError(f"MSRS metric evaluator not found: {path}")
    spec = importlib.util.spec_from_file_location("msrs_metric_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metric evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render_fused_bgr(fused: torch.Tensor, vis_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = np.clip(fused.squeeze().detach().float().cpu().numpy() * 255.0, 0, 255).round().astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _means(rows: list[dict[str, float]], prefix: str = "") -> dict[str, float]:
    if not rows:
        return {}
    return {f"{prefix}{key}": float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _output_statistics(fused: torch.Tensor) -> dict[str, float]:
    """Statistics of the floating-point luminance before rendering or quantization."""
    fused = fused.detach().float()
    local_mean = F.avg_pool2d(F.pad(fused, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
    local_second = F.avg_pool2d(F.pad(fused.square(), (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
    local_std = (local_second - local_mean.square()).clamp_min(0.0).sqrt()
    return {
        "output_mean": float(fused.mean()),
        "output_std": float(fused.std(unbiased=False)),
        "output_local_std": float(local_std.mean()),
    }


@torch.no_grad()
def evaluate_fusion_metrics(
    model: nn.Module,
    loader: DataLoader,
    names: list[str],
    device: str,
    compute_vif: bool = True,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate validation outputs with the identical render/metric convention used for test images."""
    if len(names) != len(loader.dataset):
        raise ValueError("Validation names and loader.dataset must have the same length.")
    evaluator = _evaluator()
    model.eval()
    rows: list[dict[str, float]] = []
    offset = 0
    for vis, ir in loader:
        batch_size = vis.size(0)
        fused = model(vis.to(device, non_blocking=True), ir.to(device, non_blocking=True))
        for index in range(batch_size):
            name = names[offset + index]
            vis_rgb = np.clip(vis[index].permute(1, 2, 0).numpy() * 255.0, 0, 255).round().astype(np.uint8)
            vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
            fused_bgr = _render_fused_bgr(fused[index : index + 1], vis_bgr)
            fused_gray = np.round(cv2.cvtColor(fused_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY))
            ir_gray = np.clip(ir[index, 0].numpy() * 255.0, 0, 255).round()
            vis_gray = np.round(cv2.cvtColor(vis_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY))
            row = evaluator.evaluate_triplet(
                fused_gray,
                ir_gray,
                vis_gray,
                compute_vif=compute_vif,
            )
            row.update(_output_statistics(fused[index : index + 1]))
            rows.append(row)
        offset += batch_size
    if offset != len(names):
        raise RuntimeError(f"Validated {offset} images but expected {len(names)}.")
    result = _means(rows)
    result.update(_means([row for row, name in zip(rows, names) if Path(name).stem.upper().endswith("D")], "day_"))
    result.update(_means([row for row, name in zip(rows, names) if Path(name).stem.upper().endswith("N")], "night_"))
    result["count"] = float(len(rows))
    per_image = [{"file": name, **row} for name, row in zip(names, rows)]
    return result, per_image
