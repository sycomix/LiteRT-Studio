from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from litert_studio.conversion.inspector import ModelInspection
from litert_studio.core.capabilities import (
    BackendCapabilities,
    Capability,
    SupportLevel,
)
from litert_studio.core.errors import ConfigurationError


@dataclass(frozen=True)
class ExportRequest:
    source: Path
    output: Path
    output_format: str
    quantization: str
    signatures: tuple[str, ...]
    settings: dict[str, Any]


@dataclass(frozen=True)
class ExportResult:
    artifact: Path
    manifest: dict[str, Any]


@runtime_checkable
class ArchitectureAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def capabilities(self) -> BackendCapabilities: ...

    def matches(self, inspection: ModelInspection) -> bool: ...

    def validate(self, inspection: ModelInspection, request: ExportRequest) -> tuple[str, ...]: ...

    def export(self, request: ExportRequest) -> ExportResult: ...


class LiteRTTorchGenerativeAdapter:
    """Capabilities exposed by the pinned LiteRT Torch generative exporter."""

    name = "litert-torch-generative"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            version="0.1",
            model_families=tuple(
                Capability(
                    family,
                    SupportLevel.RESEARCH,
                    "Available through a pinned LiteRT Torch model extension.",
                    ("Linux conversion environment", "Parity validation required"),
                )
                for family in ("gemma", "gemma2", "gemma3", "gemma4", "qwen3", "lfm2")
            ),
            output_formats=("litertlm",),
            quantization_policies=(
                "none",
                "dynamic_int8",
                "weight_only_int8",
                "dynamic_int4",
                "weight_only_int4",
            ),
            platforms=("android", "ios", "linux", "macos", "windows"),
        )

    def matches(self, inspection: ModelInspection) -> bool:
        return inspection.model_type in {
            "gemma",
            "gemma2",
            "gemma3",
            "gemma3_text",
            "gemma4",
            "qwen3",
            "lfm2",
        }

    def validate(self, inspection: ModelInspection, request: ExportRequest) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.matches(inspection):
            issues.append(f"{self.name} does not match model type '{inspection.model_type}'")
        if request.output_format not in self.capabilities().output_formats:
            issues.append(f"Output format '{request.output_format}' is not supported")
        if request.quantization not in self.capabilities().quantization_policies:
            issues.append(f"Quantization '{request.quantization}' is not supported")
        return tuple(issues)

    def export(self, request: ExportRequest) -> ExportResult:
        raise ConfigurationError(
            "Use the isolated LiteRT Torch export worker after compatibility preflight"
        )


class AdapterRegistry:
    def __init__(self, adapters: tuple[ArchitectureAdapter, ...] | None = None) -> None:
        self._adapters = adapters or (LiteRTTorchGenerativeAdapter(),)

    def list_capabilities(self) -> tuple[BackendCapabilities, ...]:
        return tuple(adapter.capabilities() for adapter in self._adapters)

    def resolve(self, inspection: ModelInspection) -> ArchitectureAdapter | None:
        return next((adapter for adapter in self._adapters if adapter.matches(inspection)), None)
