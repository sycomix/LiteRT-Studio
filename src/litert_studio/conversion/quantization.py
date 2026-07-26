from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from litert_studio.core.errors import ConfigurationError


@dataclass(frozen=True)
class QuantizationPolicy:
    name: str
    recipe: str
    weight_type: str
    activation_type: str
    support: str
    description: str
    constraints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


POLICIES = (
    QuantizationPolicy(
        "none",
        "",
        "source",
        "fp32",
        "baseline",
        "Unquantized parity baseline.",
    ),
    QuantizationPolicy(
        "dynamic_int8",
        "dynamic_wi8_afp32",
        "int8",
        "fp32",
        "available",
        "Dynamic-range int8 weights with fp32 activations.",
    ),
    QuantizationPolicy(
        "weight_only_int8",
        "weight_only_wi8_afp32",
        "int8",
        "fp32",
        "available",
        "Weight-only int8 with floating-point computation.",
    ),
    QuantizationPolicy(
        "dynamic_int4",
        "dynamic_wi4_afp32",
        "int4",
        "fp32",
        "experimental",
        "Dynamic int4 blockwise weights.",
        ("Weight dimensions must satisfy upstream block-size requirements.",),
    ),
    QuantizationPolicy(
        "weight_only_int4",
        "weight_only_wi4_afp32",
        "int4",
        "fp32",
        "experimental",
        "Weight-only int4 with floating-point computation.",
        ("Weight dimensions must satisfy upstream block-size requirements.",),
    ),
    QuantizationPolicy(
        "static_int8",
        "static_wi8_ai8",
        "int8",
        "int8",
        "separate_pipeline",
        "Calibration-based static int8.",
        (
            "Requires LiteRT Torch's experimental calibration workflow.",
            "Not executable by the standard export worker.",
        ),
    ),
)


def quantization_policies() -> tuple[QuantizationPolicy, ...]:
    return POLICIES


def resolve_quantization_policy(name: str) -> QuantizationPolicy:
    policy = next((item for item in POLICIES if item.name == name), None)
    if policy is None:
        allowed = ", ".join(item.name for item in POLICIES)
        raise ConfigurationError(f"Unknown quantization policy '{name}'; choose {allowed}")
    return policy


def executable_quantization_policy(name: str) -> QuantizationPolicy:
    policy = resolve_quantization_policy(name)
    if policy.support == "separate_pipeline":
        raise ConfigurationError(
            f"'{name}' requires the experimental calibration pipeline and "
            "cannot run through the standard exporter"
        )
    return policy


def policy_for_recipe(recipe: str | None) -> QuantizationPolicy:
    normalized = recipe or ""
    policy = next((item for item in POLICIES if item.recipe == normalized), None)
    if policy is None:
        raise ConfigurationError(f"Unknown LiteRT quantization recipe '{normalized}'")
    return policy
