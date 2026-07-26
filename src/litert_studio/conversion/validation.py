from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from litert_studio.core.errors import ConfigurationError


@dataclass(frozen=True)
class NumericalParity:
    passed: bool
    count: int
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    atol: float
    rtol: float
    failing_values: int

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class TokenParity:
    passed: bool
    reference_length: int
    candidate_length: int
    matching_prefix: int
    exact_match: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def compare_values(
    reference: Iterable[float],
    candidate: Iterable[float],
    *,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> NumericalParity:
    expected = tuple(float(value) for value in reference)
    actual = tuple(float(value) for value in candidate)
    if len(expected) != len(actual):
        raise ConfigurationError(
            f"Parity inputs have different lengths: {len(expected)} != {len(actual)}"
        )
    if not expected:
        raise ConfigurationError("Parity inputs must not be empty")
    if atol < 0 or rtol < 0:
        raise ConfigurationError("Parity tolerances must be non-negative")

    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    failures = 0
    for source, converted in zip(expected, actual, strict=True):
        if not math.isfinite(source) or not math.isfinite(converted):
            raise ConfigurationError("Parity inputs must contain only finite values")
        absolute = abs(source - converted)
        relative = absolute / max(abs(source), 1e-12)
        absolute_errors.append(absolute)
        relative_errors.append(relative)
        if absolute > atol + rtol * abs(source):
            failures += 1
    return NumericalParity(
        passed=failures == 0,
        count=len(expected),
        max_absolute_error=max(absolute_errors),
        mean_absolute_error=sum(absolute_errors) / len(absolute_errors),
        max_relative_error=max(relative_errors),
        atol=atol,
        rtol=rtol,
        failing_values=failures,
    )


def compare_tokens(reference: Sequence[int], candidate: Sequence[int]) -> TokenParity:
    prefix = 0
    for expected, actual in zip(reference, candidate, strict=False):
        if expected != actual:
            break
        prefix += 1
    exact = list(reference) == list(candidate)
    return TokenParity(
        passed=exact,
        reference_length=len(reference),
        candidate_length=len(candidate),
        matching_prefix=prefix,
        exact_match=exact,
    )
