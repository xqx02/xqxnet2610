from __future__ import annotations

import random
import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class PairedFusionDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, vis_dir: str | Path, ir_dir: str | Path) -> None:
        self.vis_paths = sorted(Path(vis_dir).glob("*"))
        self.ir_paths = sorted(Path(ir_dir).glob("*"))
        if len(self.vis_paths) != len(self.ir_paths):
            raise ValueError("VIS and IR folders must contain the same number of images.")

    def __len__(self) -> int:
        return len(self.vis_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        vis = Image.open(self.vis_paths[index]).convert("RGB")
        ir = Image.open(self.ir_paths[index]).convert("L")
        return to_tensor(vis), to_tensor(ir)


def _image_paths_by_stem(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    by_stem: dict[str, Path] = {}
    for path in paths:
        if path.stem in by_stem:
            raise RuntimeError(f"Duplicate image stem '{path.stem}' in {directory}")
        by_stem[path.stem] = path
    return by_stem


class OriginalFusionValidationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Original-size MSRS pairs selected by the fixed tensor split manifest."""

    def __init__(self, source_dir: str | Path, tensor_names: list[str]) -> None:
        source_dir = Path(source_dir)
        self.vi_paths = _image_paths_by_stem(source_dir / "vi")
        self.ir_paths = _image_paths_by_stem(source_dir / "ir")
        self.names = list(tensor_names)
        if len(self.names) != len(set(self.names)):
            raise ValueError("Original validation names must be unique.")
        if not self.names:
            raise ValueError("Original validation names must not be empty.")

        self.stems = [Path(name).stem for name in self.names]
        missing_vis = [stem for stem in self.stems if stem not in self.vi_paths]
        missing_ir = [stem for stem in self.stems if stem not in self.ir_paths]
        if missing_vis or missing_ir:
            missing = missing_vis[0] if missing_vis else missing_ir[0]
            modality = "VIS" if missing_vis else "IR"
            raise FileNotFoundError(f"Missing original {modality} validation image for stem '{missing}'.")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        stem = self.stems[index]
        vis = Image.open(self.vi_paths[stem]).convert("RGB")
        ir = Image.open(self.ir_paths[stem]).convert("L")
        if vis.size != ir.size:
            raise ValueError(f"Original VIS/IR dimensions differ for '{stem}': {vis.size} != {ir.size}")
        return to_tensor(vis), to_tensor(ir)


def _sanitize_vis(x: Tensor) -> Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"VIS tensor must be BCHW with 3 channels, got {tuple(x.shape)}")
    return x.squeeze(0).float().clamp(0.0, 1.0)


def _sanitize_ir(x: Tensor) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"IR tensor must be BCHW with 1 channel, got {tuple(x.shape)}")
    return x.squeeze(0).float().clamp(0.0, 1.0)


def _stratified_split_names(names: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if val_ratio <= 0:
        return sorted(names), []
    if val_ratio >= 1:
        raise ValueError("val_ratio must be smaller than 1.")

    groups: dict[str, list[str]] = {}
    for name in names:
        suffix = Path(name).stem[-1:].upper()
        groups.setdefault(suffix if suffix in {"D", "N"} else "other", []).append(name)

    target = max(1, int(len(names) * val_ratio))
    exact = {key: len(values) * target / len(names) for key, values in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    for key in sorted(groups, key=lambda item: (exact[item] - quotas[item], item), reverse=True)[: target - sum(quotas.values())]:
        quotas[key] += 1

    rng = random.Random(seed)
    train_names: list[str] = []
    val_names: list[str] = []
    for key in sorted(groups):
        values = sorted(groups[key])
        rng.shuffle(values)
        val_names.extend(values[: quotas[key]])
        train_names.extend(values[quotas[key] :])
    return sorted(train_names), sorted(val_names)


def split_tensor_names(
    tensor_dir: str | Path, val_ratio: float = 0.1, seed: int = 2026
) -> tuple[list[str], list[str]]:
    vi_dir = Path(tensor_dir) / "vi"
    names = sorted(path.name for path in vi_dir.glob("*.pt"))
    if not names:
        raise FileNotFoundError(f"No .pt tensors found in {vi_dir}")
    return _stratified_split_names(names, val_ratio, seed)


def fixed_split_tensor_names(tensor_dir: str | Path, split_file: str | Path) -> tuple[list[str], list[str], dict]:
    """Load the project's fixed MSRS validation names and validate the full partition."""
    tensor_dir = Path(tensor_dir)
    split_path = Path(split_file)
    if not split_path.is_file():
        raise FileNotFoundError(f"Fixed split manifest not found: {split_path}")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    validation_names = manifest.get("validation_names")
    if not isinstance(validation_names, list) or not all(isinstance(name, str) for name in validation_names):
        raise ValueError(f"Split manifest must contain a string list 'validation_names': {split_path}")
    available = sorted(path.name for path in (tensor_dir / "vi").glob("*.pt"))
    available_set = set(available)
    val_names = sorted(validation_names)
    if len(val_names) != len(set(val_names)):
        raise ValueError(f"Duplicate validation names in split manifest: {split_path}")
    unknown = sorted(set(val_names) - available_set)
    if unknown:
        raise FileNotFoundError(f"Validation tensor missing from {tensor_dir}: {unknown[0]}")
    train_names = sorted(available_set - set(val_names))
    if set(train_names) & set(val_names) or len(train_names) + len(val_names) != len(available):
        raise RuntimeError("Fixed split is not a disjoint full partition of tensor_dir.")

    def count(values: list[str], suffix: str) -> int:
        return sum(Path(name).stem.upper().endswith(suffix) for name in values)

    observed = {
        "train_count": len(train_names), "val_count": len(val_names),
        "train_day": count(train_names, "D"), "train_night": count(train_names, "N"),
        "val_day": count(val_names, "D"), "val_night": count(val_names, "N"),
    }
    expected = {key: manifest.get(key) for key in observed}
    if any(expected[key] is not None and expected[key] != observed[key] for key in observed):
        raise RuntimeError(f"Fixed split counts differ from manifest: expected={expected}, observed={observed}")
    resolved = {**manifest, **observed, "split_file": str(split_path.resolve()), "train_names": train_names, "val_names": val_names}
    return train_names, val_names, resolved


class TensorFusionDataset(Dataset[tuple[Tensor, Tensor]]):
    """Reads the old Fusion dataset/MSRS/train_tensor/{vi,ir} cache."""

    def __init__(self, tensor_dir: str | Path, names: list[str] | None = None) -> None:
        self.tensor_dir = Path(tensor_dir)
        self.vi_dir = self.tensor_dir / "vi"
        self.ir_dir = self.tensor_dir / "ir"
        self.names = names if names is not None else sorted(path.name for path in self.vi_dir.glob("*.pt"))
        if not self.names:
            raise FileNotFoundError(f"No .pt tensors found in {self.vi_dir}")
        missing = [name for name in self.names if not (self.ir_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing IR tensor for {missing[0]}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        name = self.names[index]
        vis = torch.load(self.vi_dir / name, map_location="cpu", weights_only=True)
        ir = torch.load(self.ir_dir / name, map_location="cpu", weights_only=True)
        return _sanitize_vis(vis), _sanitize_ir(ir)
