"""Root-finding strategy implementations."""

from __future__ import annotations

import math

from .base import RootMethod
from .models import IterationRecord, MethodResult
from .utils import PrecisionFormatter, ensure_finite


class NewtonRaphson(RootMethod):
    """Newton-Raphson method with symbolic derivative."""

    def __init__(
        self,
        *,
        function_expression: str,
        initial_guess: float,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        precision: int = 6,
        derivative_epsilon: float = 1e-14,
    ) -> None:
        super().__init__(
            function_expression=function_expression,
            name="Newton-Raphson",
            max_iterations=max_iterations,
            tolerance=tolerance,
            precision=precision,
        )
        self.initial_guess = float(initial_guess)
        self.derivative_epsilon = derivative_epsilon

    def solve(self) -> MethodResult:
        iterations: list[IterationRecord] = []
        growth_count = 0
        previous_error: float | None = None
        x_current = self.initial_guess

        for index in range(1, self.max_iterations + 1):
            f_value = float(self.function_numeric(x_current))
            derivative_value = float(self.derivative_numeric(x_current))

            if not ensure_finite(f_value, derivative_value):
                return MethodResult(
                    method_name=self.name,
                    converged=False,
                    diverged=True,
                    message="Numerical overflow/underflow detected.",
                    final_estimate=None,
                    iterations=iterations,
                )

            if abs(derivative_value) <= self.derivative_epsilon:
                iterations.append(
                    IterationRecord(
                        iteration=index,
                        estimate=PrecisionFormatter.round_scalar(x_current, self.precision),
                        error=None,
                        residual=PrecisionFormatter.round_scalar(abs(f_value), self.precision),
                    )
                )
                return MethodResult(
                    method_name=self.name,
                    converged=False,
                    diverged=True,
                    message="Zero (or near-zero) derivative encountered.",
                    final_estimate=PrecisionFormatter.round_scalar(x_current, self.precision),
                    iterations=iterations,
                )

            x_next = x_current - (f_value / derivative_value)
            if not math.isfinite(x_next):
                return MethodResult(
                    method_name=self.name,
                    converged=False,
                    diverged=True,
                    message="Method diverged to a non-finite value.",
                    final_estimate=None,
                    iterations=iterations,
                )

            error = abs(x_next - x_current)
            residual = abs(float(self.function_numeric(x_next)))
            iterations.append(
                IterationRecord(
                    iteration=index,
                    estimate=PrecisionFormatter.round_scalar(x_next, self.precision),
                    error=PrecisionFormatter.round_scalar(error, self.precision),
                    residual=PrecisionFormatter.round_scalar(residual, self.precision),
                )
            )

            if error <= self.tolerance or residual <= self.tolerance:
                return MethodResult(
                    method_name=self.name,
                    converged=True,
                    diverged=False,
                    message="Converged successfully.",
                    final_estimate=PrecisionFormatter.round_scalar(x_next, self.precision),
                    iterations=iterations,
                )

            if previous_error is not None and error > previous_error * 50:
                growth_count += 1
            else:
                growth_count = max(0, growth_count - 1)

            if growth_count >= 4:
                return MethodResult(
                    method_name=self.name,
                    converged=False,
                    diverged=True,
                    message="Divergence detected from rapidly increasing error.",
                    final_estimate=PrecisionFormatter.round_scalar(x_next, self.precision),
                    iterations=iterations,
                )

            previous_error = error
            x_current = x_next

        return MethodResult(
            method_name=self.name,
            converged=False,
            diverged=False,
            message="Maximum iterations reached without convergence.",
            final_estimate=PrecisionFormatter.round_scalar(x_current, self.precision),
            iterations=iterations,
        )


class RegulaFalsi(RootMethod):
    """Regula Falsi method with interval validation."""

    def __init__(
        self,
        *,
        function_expression: str,
        lower_bound: float,
        upper_bound: float,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        precision: int = 6,
    ) -> None:
        super().__init__(
            function_expression=function_expression,
            name="Regula Falsi",
            max_iterations=max_iterations,
            tolerance=tolerance,
            precision=precision,
        )
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self._validate_interval()

    def _validate_interval(self) -> None:
        if self.lower_bound >= self.upper_bound:
            raise ValueError("Invalid interval: lower bound must be less than upper bound.")
        f_lower = float(self.function_numeric(self.lower_bound))
        f_upper = float(self.function_numeric(self.upper_bound))
        if f_lower * f_upper >= 0:
            raise ValueError("Regula Falsi requires f(a) * f(b) < 0.")

    def solve(self) -> MethodResult:
        iterations: list[IterationRecord] = []
        stagnation_count = 0
        a = self.lower_bound
        b = self.upper_bound
        fa = float(self.function_numeric(a))
        fb = float(self.function_numeric(b))
        previous_c: float | None = None

        for index in range(1, self.max_iterations + 1):
            denominator = fb - fa
            if abs(denominator) <= 1e-18:
                return MethodResult(
                    method_name=self.name,
                    converged=False,
                    diverged=True,
                    message="Numerical instability: denominator collapsed.",
                    final_estimate=PrecisionFormatter.round_scalar(a, self.precision),
                    iterations=iterations,
                )

            c_value = ((a * fb) - (b * fa)) / denominator
            fc = float(self.function_numeric(c_value))
            if not ensure_finite(c_value, fc):
                return MethodResult(
                    method_name=self.name,
                    converged=False,
                    diverged=True,
                    message="Method diverged to non-finite values.",
                    final_estimate=None,
                    iterations=iterations,
                )

            error = None if previous_c is None else abs(c_value - previous_c)
            residual = abs(fc)
            iterations.append(
                IterationRecord(
                    iteration=index,
                    estimate=PrecisionFormatter.round_scalar(c_value, self.precision),
                    error=(
                        None
                        if error is None
                        else PrecisionFormatter.round_scalar(error, self.precision)
                    ),
                    residual=PrecisionFormatter.round_scalar(residual, self.precision),
                )
            )

            if residual <= self.tolerance or (error is not None and error <= self.tolerance):
                return MethodResult(
                    method_name=self.name,
                    converged=True,
                    diverged=False,
                    message="Converged successfully.",
                    final_estimate=PrecisionFormatter.round_scalar(c_value, self.precision),
                    iterations=iterations,
                )

            if previous_c is not None and abs(c_value - previous_c) <= 1e-15:
                stagnation_count += 1
            else:
                stagnation_count = max(0, stagnation_count - 1)

            if stagnation_count >= 5:
                return MethodResult(
                    method_name=self.name,
                    converged=False,
                    diverged=True,
                    message="Potential divergence/stagnation detected.",
                    final_estimate=PrecisionFormatter.round_scalar(c_value, self.precision),
                    iterations=iterations,
                )

            if fa * fc < 0:
                b = c_value
                fb = fc
            else:
                a = c_value
                fa = fc

            previous_c = c_value

        return MethodResult(
            method_name=self.name,
            converged=False,
            diverged=False,
            message="Maximum iterations reached without convergence.",
            final_estimate=(
                None
                if previous_c is None
                else PrecisionFormatter.round_scalar(previous_c, self.precision)
            ),
            iterations=iterations,
        )
