from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from fusion.config import load_config
from fusion.dataset import fixed_split_tensor_names
from fusion.models import build_model


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs" / "module_screening_c0_seed2026"
DUAL_BAND_RUN_ROOT = ROOT / "runs" / "e1_dualband_l2_c0_seed2026"
SOURCE_VALUE_RUN_ROOT = ROOT / "runs" / "e1_dualband_sourcev_l2_c0_seed2026"
P0_RUN = ROOT / "runs" / "loss_choice_seed2026" / "C0"
INITIAL = ROOT / "runs" / "loss_choice_seed2026" / "initial_plain_fusion_seed2026.pth"
E1_INITIAL = RUN_ROOT / "E1" / "weights" / "best_metrics.pth"
CONFIG_ROOT = ROOT / "configs" / "module_screening"
EVALUATOR = ROOT.parent / "Fusion" / "tools" / "eval_msrs_smfnet_style.py"
VALIDATION_METRICS = ("SD", "SSIM", "QABF", "VIF")
TEST_METRICS = ("EN", "SD", "SF", "MI", "VIF", "QABF", "AG", "SSIM")
RETAINED_EXPERIMENTS = ("P0", "E1")
EXPERIMENTS = {
    "P0": ("p0_plain_fusion_c0.yaml", P0_RUN, "init-weights", INITIAL),
    "E1": ("e1_conditional_csp_l1_c0.yaml", RUN_ROOT / "E1", "init-backbone", INITIAL),
    "E3": ("e3_original_rfgm_l2_c0.yaml", RUN_ROOT / "E3", "init-backbone", INITIAL),
    "F1": ("f1_e1_dualband_diff_l2_c0.yaml", DUAL_BAND_RUN_ROOT / "F1", "init-e1", E1_INITIAL),
    "F2": ("f2_e1_dualband_pred_l2_c0.yaml", DUAL_BAND_RUN_ROOT / "F2", "init-e1", E1_INITIAL),
    "F3": (
        "f3_e1_dualband_pred_sourcev_l2_c0.yaml",
        SOURCE_VALUE_RUN_ROOT / "F3",
        "init-e1",
        E1_INITIAL,
    ),
}
DEFAULT_EXPERIMENTS = ("P0", "E1", "E3")
DUAL_BAND_EXPERIMENTS = {"F1", "F2", "F3"}


def run(*arguments: object) -> None:
    command = [sys.executable, *(str(value) for value in arguments)]
    print("\n>", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparable_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["model"] = "plain_fusion"
    result["data"].pop("test_dir", None)
    return result


def load_and_verify_protocol() -> tuple[dict[str, dict], Path]:
    configs = {name: load_config(CONFIG_ROOT / item[0]) for name, item in EXPERIMENTS.items()}
    baseline = configs["P0"]
    for name, config in configs.items():
        if comparable_config(config) != comparable_config(baseline):
            raise RuntimeError(f"{name} differs from P0 outside model name/test_dir.")
    if baseline != load_config(P0_RUN / "resolved_config.yaml"):
        raise RuntimeError("The retained P0 config differs from its historical resolved config.")
    split = json.loads((ROOT / baseline["data"]["split_file"]).read_text(encoding="utf-8"))
    keys = ("train_count", "val_count", "train_day", "train_night", "val_day", "val_night")
    if tuple(split[key] for key in keys) != (975, 108, 483, 492, 53, 55):
        raise RuntimeError("The fixed day/night split has changed.")
    for source in (INITIAL, E1_INITIAL):
        if not source.is_file():
            raise FileNotFoundError(source)
    test_dirs = {(ROOT / cfg["data"]["test_dir"]).resolve() for cfg in configs.values() if cfg["data"].get("test_dir")}
    if len(test_dirs) != 1:
        raise RuntimeError("All module experiments must use one common test directory.")
    return configs, test_dirs.pop()


def verify_run(name: str, directory: Path, config: dict) -> None:
    required = (
        directory / "checkpoint.pt", directory / "weights" / "best_loss.pth",
        directory / "weights" / "best_metrics.pth", directory / "validation_selection.json",
        directory / "split.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{name} is incomplete: {missing}")
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu")
    if checkpoint.get("epoch") != 100 or checkpoint.get("config") != config:
        raise RuntimeError(f"{name} final checkpoint/config is incompatible.")
    split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    keys = ("train_count", "val_count", "train_day", "train_night", "val_day", "val_night")
    if tuple(split[key] for key in keys) != (975, 108, 483, 492, 53, 55):
        raise RuntimeError(f"{name} used an unexpected split.")
    tensor_dir = (ROOT / config["data"]["tensor_dir"]).resolve()
    split_file = (ROOT / config["data"]["split_file"]).resolve()
    expected_train, expected_val, _ = fixed_split_tensor_names(tensor_dir, split_file)
    if sorted(split.get("train_names", [])) != expected_train:
        raise RuntimeError(f"{name} used a different complete training file list.")
    if sorted(split.get("val_names", [])) != expected_val:
        raise RuntimeError(f"{name} used a different complete validation file list.")
    if name in DUAL_BAND_EXPERIMENTS:
        report_path = directory / "e1_base_init.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"{name} has no E1 initialization record.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("source_sha256") != sha256(E1_INITIAL):
            raise RuntimeError(f"{name} was initialized from a different E1 checkpoint.")
    elif name != "P0" and not (directory / "backbone_init.json").is_file():
        raise FileNotFoundError(f"{name} has no shared-backbone initialization record.")


def verify_retained_artifacts(configs: dict[str, dict]) -> None:
    manifest_path = ROOT / "runs" / "RETAINED_P0_E1_SHA256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"Retained artifact changed or is missing: {path}")
    for name in RETAINED_EXPERIMENTS:
        _, run_dir, _, _ = EXPERIMENTS[name]
        cfg = configs[name]
        verify_run(name, run_dir, cfg)
        kwargs = {
            "base_channels": int(cfg["base_channels"]),
            "encoder_blocks": tuple(cfg["encoder_blocks"]),
            "attention_heads": tuple(cfg["attention_heads"]),
            "ffn_expansion": float(cfg["ffn_expansion"]),
        }
        for kind in ("best_metrics", "best_loss"):
            state = torch.load(run_dir / "weights" / f"{kind}.pth", map_location="cpu")
            model = build_model(cfg["model"], **kwargs)
            model.load_state_dict({key.replace("module.", ""): value for key, value in state["model"].items()}, strict=True)
        for kind in ("best_metrics", "best_loss"):
            evaluation = RUN_ROOT / "evaluation" / name / kind
            images = list((evaluation / "images").glob("*.png"))
            if len(images) != 361 or not (evaluation / "metrics.csv").is_file() or not (evaluation / "done.json").is_file():
                raise RuntimeError(f"{name}/{kind} retained evaluation is incomplete.")


def train_all(configs: dict[str, dict], names: tuple[str, ...], device: str) -> None:
    for name in names:
        config_name, run_dir, initialization, source = EXPERIMENTS[name]
        config = configs[name]
        if (run_dir / "checkpoint.pt").is_file():
            verify_run(name, run_dir, config)
            print(f"Verified complete; skip training {name}.")
            continue
        latest = run_dir / "weights" / "latest.pth"
        if run_dir.exists() and any(run_dir.iterdir()) and not latest.is_file():
            raise RuntimeError(f"Partial {name} run has no resumable latest.pth: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        arguments: list[object] = [
            "train.py", "--config", CONFIG_ROOT / config_name,
            "--out", run_dir / "checkpoint.pt", "--device", device,
        ]
        arguments += ["--resume", latest] if latest.is_file() else [f"--{initialization}", source]
        run(*arguments)
        verify_run(name, run_dir, config)


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["file"] != "MEAN"]
    if len(rows) != 361:
        raise RuntimeError(f"Expected 361 metric rows: {path}")
    result = {}
    for subset, suffix, count in (("all", "", 361), ("day", "D", 179), ("night", "N", 182)):
        selected = [row for row in rows if not suffix or Path(row["file"]).stem.upper().endswith(suffix)]
        if len(selected) != count:
            raise RuntimeError(f"Unexpected {subset} count in {path}: {len(selected)}")
        result[subset] = {
            key: sum(float(row[key]) for row in selected) / count
            for key in TEST_METRICS
        }
    return result


def write_validation_summary(configs: dict[str, dict], names: tuple[str, ...], root: Path) -> None:
    rows = []
    for name in names:
        _, run_dir, _, _ = EXPERIMENTS[name]
        verify_run(name, run_dir, configs[name])
        selected = json.loads((run_dir / "validation_selection.json").read_text(encoding="utf-8"))["selected"]
        rows.append({
            "experiment": name, "epoch": selected["epoch"], "val_loss": selected["val_loss"],
            **{
                prefix + key: selected[prefix + key]
                for prefix in ("", "day_", "night_")
                for key in VALIDATION_METRICS
            },
        })
    path = root / "validation_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_root(name: str) -> Path:
    if name == "F3":
        root = SOURCE_VALUE_RUN_ROOT
    elif name in {"F1", "F2"}:
        root = DUAL_BAND_RUN_ROOT
    else:
        root = RUN_ROOT
    return root / "evaluation" / name


def evaluate_all(
    configs: dict[str, dict],
    names: tuple[str, ...],
    selected: tuple[str, ...],
    result_root: Path,
    test_dir: Path,
    device: str,
    skip_existing: bool,
) -> None:
    summaries: list[dict] = []
    baseline: dict[tuple[str, str], dict[str, float]] = {}
    selected_set = set(selected)
    for name in names:
        _, run_dir, _, _ = EXPERIMENTS[name]
        verify_run(name, run_dir, configs[name])
        for kind in ("best_metrics", "best_loss"):
            checkpoint = run_dir / "weights" / f"{kind}.pth"
            state = torch.load(checkpoint, map_location="cpu")
            epoch = int(state["epoch"])
            del state
            target = _evaluation_root(name) / kind
            images, metrics, done = target / "images", target / "metrics.csv", target / "done.json"
            context = {"checkpoint_sha256": sha256(checkpoint), "test_dir": str(test_dir)}
            if name in DUAL_BAND_EXPERIMENTS:
                split_file = ROOT / configs[name]["data"]["split_file"]
                context.update({
                    "validation_protocol": configs[name]["validation"]["protocol"],
                    "expected_pairs": 361,
                    "split_manifest_sha256": sha256(split_file),
                    "e1_source_sha256": sha256(E1_INITIAL),
                })
            is_reference = name not in selected_set
            reusable = (skip_existing or is_reference) and metrics.is_file() and done.is_file()
            if is_reference:
                if not reusable or len(list(images.glob("*.png"))) != 361:
                    raise RuntimeError(f"Retained reference evaluation is incomplete: {name}/{kind}")
            else:
                reusable = reusable and json.loads(done.read_text(encoding="utf-8")) == context
            if not reusable:
                run("infer.py", "--checkpoint", checkpoint, "--test-dir", test_dir, "--save-dir", images, "--device", device)
                run("check_results.py", "--test-dir", test_dir, "--result-dir", images, "--expected", 361)
                run(EVALUATOR, "--test_dir", test_dir, "--result_dir", images, "--out_csv", metrics)
                done.parent.mkdir(parents=True, exist_ok=True)
                done.write_text(json.dumps(context, indent=2), encoding="utf-8")
            else:
                run("check_results.py", "--test-dir", test_dir, "--result-dir", images, "--expected", 361)
            for subset, values in read_metrics(metrics).items():
                if name == "P0":
                    baseline[kind, subset] = values
                reference = baseline[kind, subset]
                summaries.append({
                    "experiment": name, "checkpoint": kind, "epoch": epoch, "subset": subset,
                    **values,
                    **{
                        f"delta_{key}": values[key] - reference[key]
                        for key in TEST_METRICS
                    },
                })
    path = result_root / "comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retained models or E1-based F1/F2/F3 under one protocol."
    )
    parser.add_argument("--stage", choices=("all", "train", "eval", "verify"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--experiments", nargs="+", choices=tuple(EXPERIMENTS))
    args = parser.parse_args()
    configs, test_dir = load_and_verify_protocol()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "verify":
        verify_retained_artifacts(configs)
        print("Retained P0/E1 weights and fused images verified; E3 remains reproducible from its config.")
        return
    selected = tuple(args.experiments or DEFAULT_EXPERIMENTS)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate experiment names are not allowed.")
    if "F3" in selected and selected != ("F3",):
        raise ValueError("F3 uses an isolated output root and must be selected alone.")
    has_dual_band = any(name in DUAL_BAND_EXPERIMENTS for name in selected)
    if has_dual_band:
        verify_retained_artifacts(configs)
    if selected == ("F3",):
        report_names = ("P0", "E1", "F1", "F2", "F3")
        result_root = SOURCE_VALUE_RUN_ROOT
        for reference in ("F1", "F2"):
            _, run_dir, _, _ = EXPERIMENTS[reference]
            verify_run(reference, run_dir, configs[reference])
            for kind in ("best_metrics", "best_loss"):
                evaluation = _evaluation_root(reference) / kind
                if (
                    len(list((evaluation / "images").glob("*.png"))) != 361
                    or not (evaluation / "metrics.csv").is_file()
                    or not (evaluation / "done.json").is_file()
                ):
                    raise RuntimeError(
                        f"Retained reference evaluation is incomplete: {reference}/{kind}"
                    )
    else:
        report_names = (
            tuple(dict.fromkeys(("P0", "E1", *selected)))
            if has_dual_band
            else selected
        )
        result_root = DUAL_BAND_RUN_ROOT if has_dual_band else RUN_ROOT
    result_root.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "train"):
        train_all(configs, selected, args.device)
    write_validation_summary(configs, report_names, result_root)
    if args.stage in ("all", "eval"):
        evaluate_all(configs, report_names, selected, result_root, test_dir, args.device, args.skip_existing)
    print(f"{','.join(selected)} requested stages completed under the fixed protocol.")


if __name__ == "__main__":
    main()
