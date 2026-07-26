from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from litert_studio.conversion import build_conversion_plan
from litert_studio.conversion.adapters import AdapterRegistry
from litert_studio.conversion.benchmark import benchmark_litert
from litert_studio.conversion.export_request import export_request_from_config
from litert_studio.conversion.litert_runtime import validate_litert_tokens
from litert_studio.conversion.reference import capture_reference_suite
from litert_studio.conversion.tensors import inspect_gemma_mapping
from litert_studio.core.config import load_versioned_config
from litert_studio.core.devices import discover_android_devices, install_android_apk
from litert_studio.core.errors import StudioError
from litert_studio.core.models import Project
from litert_studio.core.packaging import create_bundle, verify_bundle
from litert_studio.training import build_training_plan
from litert_studio.training.fixtures import fetch_smoke_model
from litert_studio.training.merge import merge_adapter
from litert_studio.training.recipe import recipe_from_config


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="litert-studio",
        description="Plan reproducible LLM fine-tuning and LiteRT conversion workflows.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Initialize a LiteRT Studio project")
    init.add_argument("name")
    init.add_argument("--directory", type=Path, default=Path.cwd())

    convert = commands.add_parser("plan-convert", help="Validate and print a conversion plan")
    convert.add_argument("config", type=Path)
    convert.add_argument("--write", type=Path, help="Also save the generated JSON plan")
    inspect_gemma = commands.add_parser(
        "inspect-gemma",
        help="Audit Gemma SafeTensors names and ranks without loading weights",
    )
    inspect_gemma.add_argument("model", type=Path)
    prepare_export = commands.add_parser(
        "prepare-export",
        help="Materialize an isolated LiteRT Torch export request",
    )
    prepare_export.add_argument("config", type=Path)
    prepare_export.add_argument("--output", type=Path, required=True)
    reference = commands.add_parser(
        "capture-reference",
        help="Capture deterministic PyTorch logits and greedy tokens",
    )
    reference.add_argument("--model", type=Path, required=True)
    reference.add_argument("--adapter", type=Path)
    reference.add_argument("--prompts", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--top-k", type=int, default=8)
    reference.add_argument("--max-new-tokens", type=int, default=8)
    merge = commands.add_parser(
        "merge-adapter",
        help="Safely merge a PEFT adapter into a full SafeTensors model",
    )
    merge.add_argument("--base-model", type=Path, required=True)
    merge.add_argument("--adapter", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    runtime = commands.add_parser(
        "validate-litert",
        help="Compare LiteRT-LM greedy tokens with a PyTorch reference suite",
    )
    runtime.add_argument("--model", type=Path, required=True)
    runtime.add_argument("--reference", type=Path, required=True)
    runtime.add_argument("--prompts", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser(
        "benchmark-litert",
        help="Measure LiteRT-LM CPU load, prefill, and decode performance",
    )
    benchmark.add_argument("--model", type=Path, required=True)
    benchmark.add_argument("--prompts", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--warmup", type=int, default=1)
    benchmark.add_argument("--iterations", type=int, default=3)
    benchmark.add_argument("--max-output-tokens", type=int, default=16)
    commands.add_parser("android-devices", help="Discover Android devices through adb")
    install_android = commands.add_parser(
        "install-android-app",
        help="Install or update the LiteRT Studio reference APK through adb",
    )
    install_android.add_argument("--apk", type=Path, required=True)
    install_android.add_argument("--serial")
    package = commands.add_parser(
        "package-artifact",
        help="Create a deterministic, checksummed release bundle",
    )
    package.add_argument("artifact_directory", type=Path)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--report", type=Path, action="append", default=[])
    verify = commands.add_parser(
        "verify-bundle",
        help="Verify every file in a LiteRT Studio release bundle",
    )
    verify.add_argument("bundle", type=Path)
    serve = commands.add_parser("serve", help="Launch the local Studio interface")
    serve.add_argument("--workspace", type=Path, default=Path.cwd())
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)

    train = commands.add_parser("plan-train", help="Validate and print a training plan")
    train.add_argument("config", type=Path)
    train.add_argument("--write", type=Path, help="Also save the generated JSON plan")
    prepare = commands.add_parser(
        "prepare-train",
        help="Materialize an isolated training-worker request",
    )
    prepare.add_argument("config", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    fixture = commands.add_parser(
        "fetch-smoke-fixture",
        help="Download the pinned tiny Gemma test model",
    )
    fixture.add_argument("--destination", type=Path, default=Path("models/tiny-random-gemma"))
    commands.add_parser("capabilities", help="List registered conversion capabilities")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            root = (args.directory / args.name).resolve()
            root.mkdir(parents=True, exist_ok=False)
            project_file = Project(args.name, root).initialize()
            print(f"Initialized {args.name} at {project_file}")
            return 0

        if args.command == "capabilities":
            import json

            capabilities = [item.to_dict() for item in AdapterRegistry().list_capabilities()]
            print(json.dumps({"conversion": capabilities}, indent=2))
            return 0
        if args.command == "fetch-smoke-fixture":
            destination = fetch_smoke_model(args.destination)
            print(f"Downloaded pinned smoke fixture to {destination}")
            return 0
        if args.command == "inspect-gemma":
            import json

            report = inspect_gemma_mapping(args.model)
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.compatible else 2
        if args.command == "capture-reference":
            suite = capture_reference_suite(
                args.model.resolve(),
                args.prompts.resolve(),
                args.output.resolve(),
                adapter_dir=args.adapter.resolve() if args.adapter else None,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"Captured {len(suite.cases)} reference cases to {args.output.resolve()}")
            return 0
        if args.command == "merge-adapter":
            result = merge_adapter(args.base_model, args.adapter, args.output)
            action = "Reused" if result.reused else "Created"
            print(f"{action} merged model at {result.model_directory}")
            return 0
        if args.command == "validate-litert":
            runtime_report = validate_litert_tokens(
                args.model,
                args.reference,
                args.prompts,
                args.output,
            )
            print(f"LiteRT token parity: {'passed' if runtime_report.passed else 'failed'}")
            return 0 if runtime_report.passed else 2
        if args.command == "benchmark-litert":
            benchmark_report = benchmark_litert(
                args.model,
                args.prompts,
                args.output,
                warmup_iterations=args.warmup,
                measured_iterations=args.iterations,
                max_output_tokens=args.max_output_tokens,
            )
            print(
                "Median decode throughput: "
                f"{benchmark_report.summary['output_tokens_per_second_median']} tokens/s"
            )
            return 0
        if args.command == "android-devices":
            import json

            print(json.dumps(discover_android_devices().to_dict(), indent=2))
            return 0
        if args.command == "install-android-app":
            import json

            print(json.dumps(install_android_apk(args.apk, args.serial), indent=2))
            return 0
        if args.command == "package-artifact":
            bundle = create_bundle(
                args.artifact_directory,
                args.output,
                reports=tuple(args.report),
            )
            print(f"Created {bundle.path} ({bundle.sha256})")
            return 0
        if args.command == "verify-bundle":
            bundle = verify_bundle(args.bundle)
            print(f"Verified {bundle.path} ({bundle.sha256})")
            return 0
        if args.command == "serve":
            if args.host not in {"127.0.0.1", "localhost", "::1"}:
                raise StudioError("The Studio interface only binds to loopback")
            try:
                import uvicorn
            except ImportError as exc:
                raise StudioError("Install LiteRT Studio with the 'api' extra") from exc
            from litert_studio.server.app import create_app

            uvicorn.run(
                create_app(args.workspace),
                host=args.host,
                port=args.port,
                log_level="info",
            )
            return 0

        expected_kind = "conversion" if args.command == "plan-convert" else "training"
        if args.command == "prepare-export":
            expected_kind = "conversion"
        config = load_versioned_config(args.config, expected_kind).values
        if args.command == "prepare-export":
            request = export_request_from_config(config)
            destination = request.write(args.output.resolve())
            print(f"Wrote LiteRT export request {request.request_id} to {destination}")
            return 0
        if args.command == "prepare-train":
            recipe = recipe_from_config(config)
            destination = recipe.write(args.output.resolve())
            print(f"Wrote training request {recipe.request_id} to {destination}")
            return 0
        plan = (
            build_conversion_plan(config)
            if args.command == "plan-convert"
            else build_training_plan(config)
        )
        rendered = plan.to_json() + "\n"
        print(rendered, end="")
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(rendered, encoding="utf-8")
        return 0
    except (StudioError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
