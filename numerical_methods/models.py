"""Datamodels shared by numerical method implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


EstimateType = Union[float, list[float]]


@dataclass(slots=True)
class IterationRecord:
    """Stores one iteration snapshot."""

    iteration: int
    estimate: EstimateType
    error: float | None = None
    residual: float | None = None
    meta: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class MethodResult:
    """Stores full solve status for a method run."""

    method_name: str
    converged: bool
    diverged: bool
    message: str
    final_estimate: EstimateType | None
    iterations: list[IterationRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[float]:
        """Returns non-null iteration errors."""
        return [value.error for value in self.iterations if value.error is not None]
