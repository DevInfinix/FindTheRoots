"""Strategy factory for solver instantiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from numerical_methods import GaussJacobi, GaussSeidel, NewtonRaphson, RegulaFalsi


@dataclass(slots=True)
class SolveRequest:
    """Normalized user input payload for solver construction."""

    method_key: str
    data: dict[str, Any]


class MethodStrategyFactory:
    """Creates solver instances based on selected strategy key."""

    @staticmethod
    def create_solver(request: SolveRequest):
        data = request.data
        method = request.method_key

        if method == "newton_raphson":
            return NewtonRaphson(
                function_expression=data["function_expression"],
                initial_guess=data["initial_guess"],
                max_iterations=data["max_iterations"],
                tolerance=data["tolerance"],
                precision=data["precision"],
            )

        if method == "regula_falsi":
            return RegulaFalsi(
                function_expression=data["function_expression"],
                lower_bound=data["lower_bound"],
                upper_bound=data["upper_bound"],
                max_iterations=data["max_iterations"],
                tolerance=data["tolerance"],
                precision=data["precision"],
            )

        if method == "gauss_jacobi":
            return GaussJacobi(
                matrix=np.array(data["matrix"], dtype=float),
                constants=np.array(data["constants"], dtype=float),
                initial_guess=np.array(data["initial_guess"], dtype=float),
                max_iterations=data["max_iterations"],
                tolerance=data["tolerance"],
                precision=data["precision"],
            )

        if method == "gauss_seidel":
            return GaussSeidel(
                matrix=np.array(data["matrix"], dtype=float),
                constants=np.array(data["constants"], dtype=float),
                initial_guess=np.array(data["initial_guess"], dtype=float),
                max_iterations=data["max_iterations"],
                tolerance=data["tolerance"],
                precision=data["precision"],
            )

        raise ValueError(f"Unsupported method key: {method}")
