"""Linear-system iterative method implementations."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .base import LinearMethod
from .models import IterationRecord, MethodResult
from .utils import PrecisionFormatter, is_diagonally_dominant


class _IterativeLinearSolver(LinearMethod):
    """Common solve loop for Jacobi and Seidel."""

    @abstractmethod
    def _next_vector(self, current: np.ndarray) -> np.ndarray:
        """Returns the next iterate vector."""

    def solve(self) -> MethodResult:
        iterations: list[IterationRecord] = []
        warnings: list[str] = []
        previous_error: float | None = None
        increasing_error_count = 0

        if not is_diagonally_dominant(self.matrix):
            warnings.append(
                "Matrix is not strictly diagonally dominant; convergence is not guaranteed."
            )

        current = self.initial_guess.astype(float)
        for index in range(1, self.max_iterations + 1):
            next_vector = self._next_vector(current.copy())
            if not np.isfinite(next_vector).all():
                return MethodResult(
                    method_name=self.name,
                    converged=False,
                    diverged=True,
                    message="Non-finite values detected during iteration.",
                    final_estimate=None,
                    iterations=iterations,
                    warnings=warnings,
                )

            error = float(np.linalg.norm(next_vector - current, ord=np.inf))
            residual = float(np.linalg.norm(self.matrix @ next_vector - self.constants, ord=np.inf))

            iterations.append(
                IterationRecord(
                    iteration=index,
                    estimate=PrecisionFormatter.round_vector(next_vector.tolist(), self.precision),
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
                    final_estimate=PrecisionFormatter.round_vector(
                        next_vector.tolist(),
                        self.precision,
                    ),
                    iterations=iterations,
                    warnings=warnings,
                )

            if previous_error is not None and error > (previous_error * 1.2):
                increasing_error_count += 1
            else:
                increasing_error_count = max(0, increasing_error_count - 1)

            if increasing_error_count >= 6:
                return MethodResult(
                    method_name=self.name,
                    converged=False,
                    diverged=True,
                    message="Divergence detected from increasing iteration error.",
                    final_estimate=PrecisionFormatter.round_vector(
                        next_vector.tolist(),
                        self.precision,
                    ),
                    iterations=iterations,
                    warnings=warnings,
                )

            current = next_vector
            previous_error = error

        return MethodResult(
            method_name=self.name,
            converged=False,
            diverged=False,
            message="Maximum iterations reached without convergence.",
            final_estimate=PrecisionFormatter.round_vector(current.tolist(), self.precision),
            iterations=iterations,
            warnings=warnings,
        )


class GaussJacobi(_IterativeLinearSolver):
    """Gauss-Jacobi: uses only previous-iteration values."""

    def __init__(
        self,
        *,
        matrix: np.ndarray,
        constants: np.ndarray,
        initial_guess: np.ndarray,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        precision: int = 6,
    ) -> None:
        super().__init__(
            matrix=matrix,
            constants=constants,
            initial_guess=initial_guess,
            name="Gauss-Jacobi",
            max_iterations=max_iterations,
            tolerance=tolerance,
            precision=precision,
        )
        diagonal = np.diag(self.matrix)
        self._inverse_diagonal = 1.0 / diagonal
        self._remainder = self.matrix - np.diagflat(diagonal)

    def _next_vector(self, current: np.ndarray) -> np.ndarray:
        return (self.constants - (self._remainder @ current)) * self._inverse_diagonal


class GaussSeidel(_IterativeLinearSolver):
    """Gauss-Seidel: uses immediate in-iteration updates."""

    def __init__(
        self,
        *,
        matrix: np.ndarray,
        constants: np.ndarray,
        initial_guess: np.ndarray,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        precision: int = 6,
    ) -> None:
        super().__init__(
            matrix=matrix,
            constants=constants,
            initial_guess=initial_guess,
            name="Gauss-Seidel",
            max_iterations=max_iterations,
            tolerance=tolerance,
            precision=precision,
        )

    def _next_vector(self, current: np.ndarray) -> np.ndarray:
        next_vector = current.copy()
        size = self.matrix.shape[0]
        for row in range(size):
            left_sum = np.dot(self.matrix[row, :row], next_vector[:row])
            right_sum = np.dot(self.matrix[row, row + 1 :], current[row + 1 :])
            next_vector[row] = (self.constants[row] - left_sum - right_sum) / self.matrix[row, row]
        return next_vector
