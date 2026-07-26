from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class SupportLevel(str, Enum):
    UNSUPPORTED = "unsupported"
    INSPECTION = "inspection"
    RESEARCH = "research"
    EXPERIMENTAL = "experimental"
    SUPPORTED = "supported"


@dataclass(frozen=True)
class Capability:
    name: str
    support: SupportLevel
    description: str
    constraints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["support"] = self.support.value
        return value


@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    version: str
    model_families: tuple[Capability, ...]
    output_formats: tuple[str, ...]
    quantization_policies: tuple[str, ...]
    platforms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "version": self.version,
            "model_families": [item.to_dict() for item in self.model_families],
            "output_formats": list(self.output_formats),
            "quantization_policies": list(self.quantization_policies),
            "platforms": list(self.platforms),
        }
