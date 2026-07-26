from __future__ import annotations

from pathlib import Path

from litert_studio.conversion.adapters import AdapterRegistry
from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.core.capabilities import SupportLevel


def test_gemma_research_adapter_is_explicit(model_dir: Path) -> None:
    config = model_dir / "config.json"
    config.write_text(
        '{"model_type": "gemma3", "architectures": ["Gemma3ForCausalLM"]}',
        encoding="utf-8",
    )
    adapter = AdapterRegistry().resolve(inspect_model_directory(model_dir))
    assert adapter is not None
    capability = adapter.capabilities()
    assert capability.output_formats == ("litertlm",)
    assert capability.model_families[0].support is SupportLevel.RESEARCH


def test_unknown_model_has_no_adapter(model_dir: Path) -> None:
    assert AdapterRegistry().resolve(inspect_model_directory(model_dir)) is None
