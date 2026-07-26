from __future__ import annotations

from pathlib import Path
from typing import Any

from litert_studio.conversion.adapters import AdapterRegistry
from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.conversion.quantization import executable_quantization_policy
from litert_studio.core.errors import ConfigurationError
from litert_studio.core.models import JobKind, JobPlan, Stage


def build_conversion_plan(config: dict[str, Any]) -> JobPlan:
    source = Path(_string(config, "source"))
    output = Path(_string(config, "output"))
    quantization = str(config.get("quantization", "dynamic_int8"))
    policy = executable_quantization_policy(quantization)

    inspection = inspect_model_directory(source)
    registry_adapter = AdapterRegistry().resolve(inspection)
    adapter = str(
        config.get(
            "adapter",
            registry_adapter.name if registry_adapter else f"{inspection.model_type}-adapter",
        )
    )
    output_format = str(config.get("output_format", "litertlm" if registry_adapter else "tflite"))
    signatures = config.get("signatures", ["prefill", "decode"])
    if not isinstance(signatures, list) or not all(isinstance(item, str) for item in signatures):
        raise ConfigurationError("'signatures' must be a list of strings")

    warnings: list[str] = [
        "No architecture adapter has been executed; this is a dry-run plan.",
        "Compatibility must be established by numerical and on-device validation.",
    ]
    if registry_adapter:
        capabilities = registry_adapter.capabilities()
        support = capabilities.model_families[0].support.value
        warnings.append(f"Matched {adapter} with capability level '{support}'.")
        if output_format not in capabilities.output_formats:
            warnings.append(
                f"{adapter} does not declare output format '{output_format}' as compatible."
            )
        if quantization not in capabilities.quantization_policies:
            warnings.append(
                f"{adapter} does not declare quantization '{quantization}' as compatible."
            )
    else:
        warnings.append(f"No registered adapter matches model type '{inspection.model_type}'.")
    calibration = config.get("representative_dataset")
    if quantization in {"dynamic_int4", "weight_only_int4"}:
        warnings.extend(policy.constraints)
    if not inspection.tokenizer_assets:
        warnings.append("No tokenizer assets were found beside the checkpoint.")

    export_backend = "litert-torch-generative" if registry_adapter else "tensorflow-savedmodel"
    converter_backend = "litert-torch" if registry_adapter else "litert-converter"
    stages = (
        Stage("inspect", "Validate checkpoint layout and provenance", "studio-core"),
        Stage("load", "Load SafeTensors and validate tensor schema", adapter),
        Stage("transform", "Map source tensors into an exportable graph", adapter),
        Stage(
            "export",
            "Export explicit inference signatures",
            export_backend,
            {"signatures": signatures},
        ),
        Stage(
            "convert",
            "Convert the graph to a LiteRT flatbuffer",
            converter_backend,
            {"operator_policy": config.get("operator_policy", "builtins-preferred")},
        ),
        Stage(
            "quantize",
            "Apply the selected weight/activation policy",
            converter_backend,
            {
                "policy": quantization,
                "recipe": policy.recipe,
                "representative_dataset": calibration,
            },
        ),
        Stage(
            "verify",
            "Compare reference logits and token generation",
            "studio-validator",
            {"atol": config.get("atol", 0.01), "rtol": config.get("rtol", 0.01)},
        ),
        Stage(
            "package",
            "Write model, assets, report, and manifest",
            "litertlm-packager" if output_format == "litertlm" else "studio-packager",
        ),
    )
    return JobPlan(
        kind=JobKind.CONVERSION,
        name=str(config.get("name", f"convert-{inspection.model_type}")),
        stages=stages,
        inputs={"model": inspection.to_dict()},
        outputs={"directory": str(output.resolve()), "format": output_format},
        warnings=tuple(warnings),
    )


def _string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{key}' must be a non-empty string")
    return value
