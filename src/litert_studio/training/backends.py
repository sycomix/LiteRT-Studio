from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from litert_studio.core.capabilities import (
    BackendCapabilities,
    Capability,
    SupportLevel,
)


@dataclass(frozen=True)
class TrainingRequest:
    base_model: Path
    dataset: Path
    output: Path
    method: str
    settings: dict[str, Any]


@runtime_checkable
class TrainingBackend(Protocol):
    @property
    def name(self) -> str: ...

    def capabilities(self) -> BackendCapabilities: ...

    def validate(self, request: TrainingRequest) -> tuple[str, ...]: ...

    def command(self, request: TrainingRequest) -> tuple[str, ...]: ...


class TransformersPeftBackend:
    name = "transformers-peft"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            version="0.1",
            model_families=(
                Capability(
                    "transformers-causal-lm",
                    SupportLevel.EXPERIMENTAL,
                    "LoRA and QLoRA recipes using Transformers and PEFT.",
                ),
            ),
            output_formats=("adapter-safetensors", "merged-safetensors"),
            quantization_policies=("qlora-nf4",),
            platforms=("linux", "windows"),
        )

    def validate(self, request: TrainingRequest) -> tuple[str, ...]:
        issues: list[str] = []
        if request.method not in {"lora", "qlora"}:
            issues.append("The initial Transformers/PEFT backend supports only LoRA and QLoRA")
        if not request.dataset.is_file():
            issues.append(f"Dataset does not exist: {request.dataset}")
        if not request.base_model.is_dir():
            issues.append(f"Base model does not exist: {request.base_model}")
        return tuple(issues)

    def command(self, request: TrainingRequest) -> tuple[str, ...]:
        # Execution lands after the recipe module is isolated as a subprocess entry point.
        return (
            "python",
            "-m",
            "litert_studio.training.worker",
            "--request",
            "<materialized-request.json>",
            "--execute",
        )
