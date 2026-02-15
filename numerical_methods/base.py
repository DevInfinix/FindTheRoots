"""Abstract class hierarchy for numerical methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from .models import MethodResult
from .utils import FunctionParser


class BaseMethod(ABC):
    """Common config and contract for all numerical strategies."""

    def __init__(
        self,
        *,
        name: str,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        precision: int = 6,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if precision < 0:
            raise ValueError("precision must be non-negative.")

        self.name = name
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.precision = precision

    @abstractmethod
    def solve(self) -> MethodResult:
        """Runs the selected method."""


class RootMethod(BaseMethod):
    """Shared behavior for scalar root-finding methods."""

    def __init__(
        self,
        *,
        function_expression: str,
        name: str,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        precision: int = 6,
    ) -> None:
        super().__init__(
            name=name,
            max_iterations=max_iterations,
            tolerance=tolerance,
            precision=precision,
        )
        self.function_expression = function_expression
        self.x_symbol, self.function_expr = FunctionParser.parse(function_expression)
        self.derivative_expr = sp.diff(self.function_expr, self.x_symbol)
        self.function_numeric = sp.lambdify(
            self.x_symbol,
            self.function_expr,
            modules="numpy",
        )
        self.derivative_numeric = sp.lambdify(
            self.x_symbol,
            self.derivative_expr,
            modules="numpy",
        )


class LinearMethod(BaseMethod):
    """Shared behavior for linear-system iterative solvers."""

    def __init__(
        self,
        *,
        matrix: np.ndarray,
        constants: np.ndarray,
        initial_guess: np.ndarray,
        name: str,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        precision: int = 6,
    ) -> None:
        super().__init__(
            name=name,
            max_iterations=max_iterations,
            tolerance=tolerance,
            precision=precision,
        )
        self.matrix = np.array(matrix, dtype=float)
        self.constants = np.array(constants, dtype=float)
        self.initial_guess = np.array(initial_guess, dtype=float)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("Coefficient matrix must be 2-dimensional.")
        rows, cols = self.matrix.shape
        if rows != cols:
            raise ValueError("Coefficient matrix must be square.")
        if self.constants.shape != (rows,):
            raise ValueError("Constant vector size must match matrix size.")
        if self.initial_guess.shape != (rows,):
            raise ValueError("Initial guess size must match matrix size.")
        if np.any(np.isclose(np.diag(self.matrix), 0.0)):
            raise ValueError("Matrix contains zero diagonal entries.")
