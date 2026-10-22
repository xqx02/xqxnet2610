from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _index_by_stem(directory: Path, label: str) -> Dict[str, Path]:
    if not directory.is_dir():
        raise SystemExit("{} directory not found: {}".format(label, directory))
    indexed: Dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.stem in indexed:
            raise SystemExit(
                "Duplicate {} stem {!r}: {} and {}".format(
                    label, path.stem, indexed[path.stem], path
                )
            )
        indexed[path.stem] = path
    return indexed


def _validate_images(paths: Dict[str, Path], label: str, expected_size) -> None:
    for stem, path in paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SystemExit("Could not decode {} image: {}".format(label, path))
        size = image.shape[1], image.shape[0]
        if size != expected_size:
            raise SystemExit(
                "Invalid {} size for {}: found {}, expected {}.".format(
                    label, stem, size, expected_size
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly validate fixed MSRS source pairs and fused outputs."
    )
    parser.add_argument("--test-dir", required=True, help="Dataset test directory containing vi/ and ir/.")
    parser.add_argument("--result-dir", required=True, help="Directory containing fused result images.")
    parser.add_argument("--expected", type=int, default=361)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_dir = Path(args.test_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    expected_size = args.width, args.height

    vi_by_stem = _index_by_stem(test_dir / "vi", "VIS")
    ir_by_stem = _index_by_stem(test_dir / "ir", "IR")
    result_by_stem = _index_by_stem(result_dir, "result")

    if len(vi_by_stem) != args.expected or len(ir_by_stem) != args.expected:
        raise SystemExit(
            "Expected {} source pairs, found VIS={} and IR={}.".format(
                args.expected, len(vi_by_stem), len(ir_by_stem)
            )
        )
    if len(result_by_stem) != args.expected:
        raise SystemExit(
            "Expected {} fused results, found {}.".format(args.expected, len(result_by_stem))
        )

    vi_stems = set(vi_by_stem)
    ir_stems = set(ir_by_stem)
    result_stems = set(result_by_stem)
    missing_ir = sorted(vi_stems - ir_stems)
    extra_ir = sorted(ir_stems - vi_stems)
    missing_results = sorted(vi_stems - result_stems)
    extra_results = sorted(result_stems - vi_stems)
    if missing_ir or extra_ir:
        raise SystemExit(
            "IR/VIS filename mismatch: missing IR={}, extra IR={}.".format(
                missing_ir[:3], extra_ir[:3]
            )
        )
    if missing_results or extra_results:
        raise SystemExit(
            "Result/VIS filename mismatch: missing results={}, extra results={}.".format(
                missing_results[:3], extra_results[:3]
            )
        )

    _validate_images(vi_by_stem, "VIS", expected_size)
    _validate_images(ir_by_stem, "IR", expected_size)
    _validate_images(result_by_stem, "result", expected_size)
    print(
        "OK: {} aligned MSRS pairs and {} fused results; all decode at {}x{}.".format(
            len(vi_by_stem), len(result_by_stem), args.width, args.height
        )
    )


if __name__ == "__main__":
    main()
