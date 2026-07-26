from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from litert_studio.conversion.quantization import (
    executable_quantization_policy,
    policy_for_recipe,
)
from litert_studio.core.errors import ConfigurationError


@dataclass(frozen=True)
class LiteRTExportRequest:
    model: str
    output_dir: str
    prefill_lengths: tuple[int, ...] = (128,)
    cache_length: int = 1024
    quantization_recipe: str | None = "dynamic_wi8_afp32"
    externalize_embedder: bool = False
    use_jinja_template: bool = False
    bundle_litert_lm: bool = True
    task: str = "text_generation"
    trust_remote_code: bool = False
    schema_version: str = "1"

    @property
    def request_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["prefill_lengths"] = list(self.prefill_lengths)
        value["request_id"] = self.request_id
        return value

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def export_request_from_config(config: dict[str, Any]) -> LiteRTExportRequest:
    prefill = config.get("prefill_lengths", [128])
    if not isinstance(prefill, list) or not prefill:
        raise ConfigurationError("'prefill_lengths' must be a non-empty list")
    lengths = tuple(_positive_int(value, "prefill_lengths") for value in prefill)
    requested_recipe = config.get("quantization_recipe")
    quantization = config.get("quantization")
    if quantization is None and isinstance(requested_recipe, str):
        policy = policy_for_recipe(requested_recipe)
    else:
        policy = executable_quantization_policy(str(quantization or "dynamic_int8"))
    if requested_recipe is not None and requested_recipe != policy.recipe:
        raise ConfigurationError(
            f"Recipe '{requested_recipe}' does not match policy '{policy.name}' "
            f"(expected '{policy.recipe}')"
        )
    return LiteRTExportRequest(
        model=str(Path(_string(config, "source")).resolve()),
        output_dir=str(Path(_string(config, "output")).resolve()),
        prefill_lengths=lengths,
        cache_length=_positive_int(config.get("cache_length", 1024), "cache_length"),
        quantization_recipe=policy.recipe,
        externalize_embedder=bool(config.get("externalize_embedder", False)),
        use_jinja_template=bool(config.get("use_jinja_template", False)),
        bundle_litert_lm=bool(config.get("bundle_litert_lm", True)),
        trust_remote_code=False,
    )


def export_request_from_file(path: Path) -> LiteRTExportRequest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Invalid LiteRT export request: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("LiteRT export request must be an object")
    translated = {
        "source": data.get("model"),
        "output": data.get("output_dir"),
        **data,
    }
    return export_request_from_config(translated)


def _string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{key}' must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"'{name}' values must be positive integers")
    return value
