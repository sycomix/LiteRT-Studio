from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from litert_studio.conversion.quantization import policy_for_recipe
from litert_studio.conversion.reference import load_prompts
from litert_studio.core.errors import ConfigurationError
from litert_studio.core.manifest import describe_artifact


@dataclass(frozen=True)
class BenchmarkSample:
    prompt_index: int
    iteration: int
    prefill_ms: float
    decode_ms: float
    output_tokens: int
    output_tokens_per_second: float


@dataclass(frozen=True)
class LiteRTBenchmarkReport:
    model_sha256: str
    quantization: str
    backend: str
    device: str
    litert_lm_version: str
    load_ms: float
    warmup_iterations: int
    measured_iterations: int
    max_output_tokens: int
    samples: tuple[BenchmarkSample, ...]
    summary: dict[str, float]
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def benchmark_litert(
    model: Path,
    prompts_path: Path,
    output: Path,
    *,
    warmup_iterations: int = 1,
    measured_iterations: int = 3,
    max_output_tokens: int = 16,
) -> LiteRTBenchmarkReport:
    if warmup_iterations < 0:
        raise ConfigurationError("Warmup iterations cannot be negative")
    if measured_iterations <= 0 or max_output_tokens <= 0:
        raise ConfigurationError("Measured iterations and output tokens must be positive")
    prompts = load_prompts(prompts_path)
    try:
        from litert_lm.engine import Engine  # type: ignore[import-not-found,import-untyped]
        from litert_lm.interfaces import (  # type: ignore[import-not-found,import-untyped]
            Backend,
            SamplerConfig,
        )
    except ImportError as exc:
        raise ConfigurationError("Install litert-lm to run benchmarks") from exc

    load_started = time.perf_counter()
    engine = Engine(str(model), backend=Backend.CPU(), max_num_tokens=2048)
    load_ms = _milliseconds(load_started)
    samples: list[BenchmarkSample] = []
    try:
        for iteration in range(warmup_iterations + measured_iterations):
            for prompt_index, prompt in enumerate(prompts):
                with engine.create_session(
                    apply_prompt_template=False,
                    sampler_config=SamplerConfig(top_k=1, temperature=0, seed=42),
                    max_output_tokens=max_output_tokens,
                ) as session:
                    prefill_started = time.perf_counter()
                    session.run_prefill([prompt])
                    prefill_ms = _milliseconds(prefill_started)
                    decode_started = time.perf_counter()
                    response = session.run_decode()
                    decode_ms = _milliseconds(decode_started)
                if iteration < warmup_iterations:
                    continue
                text = response.texts[0] if response.texts else ""
                output_tokens = len(engine.tokenize(text))
                samples.append(
                    BenchmarkSample(
                        prompt_index=prompt_index,
                        iteration=iteration - warmup_iterations,
                        prefill_ms=round(prefill_ms, 3),
                        decode_ms=round(decode_ms, 3),
                        output_tokens=output_tokens,
                        output_tokens_per_second=round(
                            output_tokens / max(decode_ms / 1000, 1e-9), 3
                        ),
                    )
                )
    finally:
        close = getattr(engine, "close", None)
        if close is not None:
            close()
    report = LiteRTBenchmarkReport(
        model_sha256=describe_artifact(model).sha256,
        quantization=_quantization_for_model(model),
        backend="cpu",
        device=f"{platform.system()} {platform.machine()}",
        litert_lm_version=version("litert-lm"),
        load_ms=round(load_ms, 3),
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        max_output_tokens=max_output_tokens,
        samples=tuple(samples),
        summary={
            "prefill_ms_median": round(statistics.median(item.prefill_ms for item in samples), 3),
            "decode_ms_median": round(statistics.median(item.decode_ms for item in samples), 3),
            "output_tokens_per_second_median": round(
                statistics.median(item.output_tokens_per_second for item in samples),
                3,
            ),
        },
    )
    report.write(output)
    return report


def _milliseconds(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _quantization_for_model(model: Path) -> str:
    manifest_path = model.resolve().parent / "manifest.json"
    if not manifest_path.is_file():
        return "unknown"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recipe = manifest["settings"]["quantization_recipe"]
        return policy_for_recipe(recipe).name
    except (json.JSONDecodeError, KeyError, TypeError, ConfigurationError):
        return "unknown"
