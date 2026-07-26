from __future__ import annotations

import pytest

from litert_studio.conversion.validation import compare_tokens, compare_values
from litert_studio.core.errors import ConfigurationError


def test_numerical_parity_uses_absolute_and_relative_tolerance() -> None:
    report = compare_values([1.0, 2.0, 3.0], [1.001, 2.002, 3.003])
    assert report.passed
    assert report.failing_values == 0
    assert report.max_absolute_error == pytest.approx(0.003)


def test_numerical_parity_reports_failure() -> None:
    report = compare_values([1.0, 2.0], [1.0, 3.0], atol=0.01, rtol=0)
    assert not report.passed
    assert report.failing_values == 1


def test_numerical_parity_rejects_shape_mismatch() -> None:
    with pytest.raises(ConfigurationError, match="different lengths"):
        compare_values([1.0], [1.0, 2.0])


def test_token_parity_reports_matching_prefix() -> None:
    report = compare_tokens([1, 2, 3], [1, 2, 4])
    assert not report.passed
    assert report.matching_prefix == 2
