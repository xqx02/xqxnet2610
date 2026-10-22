from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from fusion.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vis", default=None)
    parser.add_argument("--ir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _image_names(directory: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in exts)


def _find_by_stem(directory: Path, stem: str) -> Path | None:
    for path in _image_names(directory):
        if path.stem == stem:
            return path
    return None


def _run_one(model, vis_path: Path, ir_path: Path, out_path: Path, device: str) -> None:
    vis_bgr = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
    ir_gray = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    if vis_bgr is None:
        raise FileNotFoundError(f"Failed to read visible image: {vis_path}")
    if ir_gray is None:
        raise FileNotFoundError(f"Failed to read infrared image: {ir_path}")
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis = torch.from_numpy(vis_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    ir = torch.from_numpy(ir_gray).unsqueeze(0).unsqueeze(0).float().div(255.0).to(device)
    with torch.no_grad():
        fused_y = model(vis, ir).squeeze(0).squeeze(0).detach().cpu().numpy()
    ycrcb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = np.clip(fused_y * 255.0, 0, 255).round().astype(np.uint8)
    out_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_bgr)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    model = build_model(
        cfg["model"],
        base_channels=int(cfg["base_channels"]),
        encoder_blocks=tuple(cfg["encoder_blocks"]),
        attention_heads=tuple(cfg.get("attention_heads", [1, 2, 4, 8])),
        ffn_expansion=float(cfg.get("ffn_expansion", 2.66)),
    ).to(args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    if args.test_dir:
        test_dir = Path(args.test_dir)
        save_dir = Path(args.save_dir or "output/msrs_clean")
        vi_dir = test_dir / "vi"
        ir_dir = test_dir / "ir"
        written = 0
        missing: list[str] = []
        for vis_path in _image_names(vi_dir):
            ir_path = _find_by_stem(ir_dir, vis_path.stem)
            if ir_path is not None:
                _run_one(model, vis_path, ir_path, save_dir / f"{vis_path.stem}.png", args.device)
                written += 1
            else:
                missing.append(vis_path.name)
        if missing:
            raise RuntimeError(f"Missing {len(missing)} IR pairs. First missing: {missing[0]}")
        print(f"Wrote {written} fused images to {save_dir}")
        return

    if not (args.vis and args.ir and args.out):
        raise ValueError("Use either --test-dir/--save-dir or --vis/--ir/--out.")
    _run_one(model, Path(args.vis), Path(args.ir), Path(args.out), args.device)


if __name__ == "__main__":
    main()
