from __future__ import annotations

import pytest

from litert_studio.conversion.export_request import export_request_from_config
from litert_studio.conversion.quantization import (
    executable_quantization_policy,
    policy_for_recipe,
    quantization_policies,
)
from litert_studio.core.errors import ConfigurationError


def test_policy_registry_uses_pinned_exporter_recipe_names() -> None:
    recipes = {policy.recipe for policy in quantization_policies()}

    assert "" in recipes
    assert "dynamic_wi8_afp32" in recipes
    assert "weight_only_wi8_afp32" in recipes
    assert "dynamic_wi4_afp32" in recipes
    assert "weight_only_wi4_afp32" in recipes


def test_export_request_translates_policy_to_recipe(tmp_path) -> None:
    request = export_request_from_config(
        {
            "source": str(tmp_path / "model"),
            "output": str(tmp_path / "output"),
            "quantization": "weight_only_int8",
        }
    )

    assert request.quantization_recipe == "weight_only_wi8_afp32"
    assert policy_for_recipe(request.quantization_recipe).name == "weight_only_int8"


def test_mismatched_raw_recipe_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="does not match"):
        export_request_from_config(
            {
                "source": str(tmp_path / "model"),
                "output": str(tmp_path / "output"),
                "quantization": "none",
                "quantization_recipe": "dynamic_wi8_afp32",
            }
        )


def test_static_int8_is_not_sent_to_standard_exporter() -> None:
    with pytest.raises(ConfigurationError, match="experimental calibration"):
        executable_quantization_policy("static_int8")
