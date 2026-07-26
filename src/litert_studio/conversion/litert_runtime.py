from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from litert_studio.conversion.reference import load_prompts
from litert_studio.conversion.validation import TokenParity, compare_tokens
from litert_studio.core.errors import ConfigurationError
from litert_studio.core.manifest import describe_artifact


@dataclass(frozen=True)
class RuntimeCase:
    prompt_sha256: str
    input_tokens_match: bool
    token_parity: TokenParity
    candidate_ids: tuple[int, ...]


@dataclass(frozen=True)
class LiteRTRuntimeReport:
    passed: bool
    model_sha256: str
    litert_lm_version: str
    backend: str
    logits_available: bool
    cases: tuple[RuntimeCase, ...]
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def validate_litert_tokens(
    model: Path,
    reference_path: Path,
    prompts_path: Path,
    output: Path,
) -> LiteRTRuntimeReport:
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_cases = reference["cases"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigurationError(f"Invalid reference suite: {reference_path}") from exc
    prompts = load_prompts(prompts_path)
    if len(prompts) != len(reference_cases):
        raise ConfigurationError("Prompt and reference case counts differ")

    try:
        from litert_lm.engine import Engine  # type: ignore[import-not-found]
        from litert_lm.interfaces import (  # type: ignore[import-not-found]
            Backend,
            SamplerConfig,
        )
    except ImportError as exc:
        raise ConfigurationError("Install litert-lm to run runtime validation") from exc

    results: list[RuntimeCase] = []
    with Engine(str(model), backend=Backend.CPU(), max_num_tokens=128) as engine:
        for prompt, case in zip(prompts, reference_cases, strict=True):
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt_hash != case.get("prompt_sha256"):
                raise ConfigurationError("Prompt content does not match the reference suite")
            expected_input = [int(value) for value in case["input_ids"]]
            runtime_input = engine.tokenize(prompt)
            if engine.bos_token_id is not None:
                runtime_input = [engine.bos_token_id, *runtime_input]
            input_match = runtime_input == expected_input

            with engine.create_session(
                apply_prompt_template=False,
                sampler_config=SamplerConfig(top_k=1, temperature=0, seed=42),
            ) as session:
                session.run_prefill([prompt])
                response = session.run_decode()
            expected_generated = [int(value) for value in case["generated_ids"]]
            response_text = response.texts[0] if response.texts else ""
            candidate = engine.tokenize(response_text)[: len(expected_generated)]
            parity = compare_tokens(expected_generated, candidate)
            results.append(
                RuntimeCase(
                    prompt_sha256=prompt_hash,
                    input_tokens_match=input_match,
                    token_parity=parity,
                    candidate_ids=tuple(candidate),
                )
            )
    report = LiteRTRuntimeReport(
        passed=all(item.input_tokens_match and item.token_parity.passed for item in results),
        model_sha256=describe_artifact(model).sha256,
        litert_lm_version=version("litert-lm"),
        backend="cpu",
        logits_available=False,
        cases=tuple(results),
    )
    report.write(output)
    _record_runtime_result(model, output, report)
    return report


def _record_runtime_result(
    model: Path,
    report_path: Path,
    report: LiteRTRuntimeReport,
) -> None:
    manifest_path = model.resolve().parent / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = manifest["validation"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return
    if not isinstance(validation, dict):
        return
    validation["runtime"] = "passed" if report.passed else "failed"
    validation["runtime_report"] = {
        "path": str(report_path.resolve()),
        "sha256": describe_artifact(report_path).sha256,
        "backend": report.backend,
        "litert_lm_version": report.litert_lm_version,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
