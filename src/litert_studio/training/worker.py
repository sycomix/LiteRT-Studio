from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from litert_studio.core.errors import ConfigurationError
from litert_studio.training.datasets import inspect_jsonl
from litert_studio.training.recipe import TrainingRecipe, recipe_from_file


@dataclass(frozen=True)
class PreflightReport:
    ready: bool
    request_id: str
    packages: dict[str, str | None]
    accelerator: str
    issues: tuple[str, ...]
    dataset_fingerprint: str
    eval_dataset_fingerprint: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def preflight(recipe: TrainingRecipe) -> PreflightReport:
    issues: list[str] = []
    base_model = Path(recipe.base_model)
    dataset = Path(recipe.dataset)
    eval_dataset = Path(recipe.eval_dataset) if recipe.eval_dataset is not None else None
    output = Path(recipe.output)
    if not base_model.is_dir():
        issues.append(f"Base model does not exist: {base_model}")
    try:
        dataset_inspection = inspect_jsonl(dataset)
        fingerprint = dataset_inspection.fingerprint
    except ConfigurationError as exc:
        issues.append(str(exc))
        fingerprint = "unavailable"
    eval_fingerprint = None
    if eval_dataset is not None:
        try:
            eval_fingerprint = inspect_jsonl(eval_dataset).fingerprint
        except ConfigurationError as exc:
            issues.append(str(exc))
            eval_fingerprint = "unavailable"
    if output in {base_model, dataset, eval_dataset}:
        issues.append("Output must not overwrite the base model or a dataset")
    if recipe.method not in {"lora", "qlora", "full"}:
        issues.append("Training method must be LoRA, QLoRA, or full")
    if not 0 <= recipe.lora.dropout <= 1:
        issues.append("LoRA dropout must be between 0 and 1")
    if recipe.epochs <= 0 or recipe.learning_rate <= 0:
        issues.append("Epochs and learning rate must be positive")

    package_names = ["torch", "transformers", "peft", "safetensors", "accelerate"]
    if recipe.method == "qlora":
        package_names.append("bitsandbytes")
    packages = {name: _package_version(name) for name in package_names}
    missing = [name for name, found in packages.items() if found is None]
    if missing:
        issues.append(f"Missing training packages: {', '.join(missing)}")
    accelerator = "unavailable"
    if packages.get("torch") is not None:
        accelerator, runtime_issue = _probe_torch_runtime()
        if runtime_issue is not None:
            issues.append(runtime_issue)
    return PreflightReport(
        ready=not issues,
        request_id=recipe.request_id,
        packages=packages,
        accelerator=accelerator,
        issues=tuple(issues),
        dataset_fingerprint=fingerprint,
        eval_dataset_fingerprint=eval_fingerprint,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Isolated LiteRT Studio training worker")
    root.add_argument("--request", type=Path, required=True)
    root.add_argument(
        "--preflight",
        action="store_true",
        help="Validate inputs and dependencies without loading model weights",
    )
    root.add_argument(
        "--execute",
        action="store_true",
        help="Run the optional Transformers/PEFT backend after preflight",
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        recipe = recipe_from_file(args.request)
        report = preflight(recipe)
        print(json.dumps(report.to_dict(), indent=2))
        if args.preflight or not args.execute:
            return 0 if report.ready else 2
        if not report.ready:
            print("error: training preflight failed", file=sys.stderr)
            return 2
        from litert_studio.training.transformers_backend import run_transformers_peft

        result = run_transformers_peft(recipe)
        print(
            json.dumps(
                {
                    "status": "succeeded",
                    "adapter_directory": str(result.adapter_directory),
                    "manifest": str(result.manifest),
                    "metrics": result.metrics,
                    "reused": result.reused,
                },
                indent=2,
            )
        )
        return 0
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _probe_torch_runtime() -> tuple[str, str | None]:
    """Import PyTorch out-of-process so a broken native runtime cannot crash Studio."""
    script = (
        "import json, torch; "
        "print(json.dumps({'cuda': torch.cuda.is_available(), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'torch': torch.__version__, 'cuda_runtime': torch.version.cuda}))"
    )
    environment = dict(os.environ)
    environment.setdefault("USE_TF", "0")
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "unavailable", f"PyTorch runtime probe failed: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        final_line = detail[-1] if detail else f"process exited with status {result.returncode}"
        if "c10.dll" in (result.stderr + result.stdout):
            final_line = (
                "PyTorch cannot load c10.dll or one of its native dependencies. "
                "Reinstall an official Windows CPU/CUDA wheel and verify the Microsoft "
                "Visual C++ runtime."
            )
        return "unavailable", final_line
    try:
        details = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return "unavailable", "PyTorch runtime probe returned an unreadable result"
    if details.get("cuda"):
        return f"cuda: {details.get('device') or 'available'}", None
    return "cpu", None


if __name__ == "__main__":
    raise SystemExit(main())
