"""Utility helpers for precision and validation."""

from __future__ import annotations

import math

import numpy as np
import sympy as sp


class PrecisionFormatter:
    """Utility to round and format values using user-selected precision."""

    @staticmethod
    def round_scalar(value: float, precision: int) -> float:
        rounded = round(float(value), precision)
        # Avoid displaying "-0.0" from tiny signed floating artifacts.
        if abs(rounded) < 10 ** (-(precision + 1)):
            return 0.0
        return rounded

    @staticmethod
    def round_vector(values: list[float], precision: int) -> list[float]:
        return [PrecisionFormatter.round_scalar(float(item), precision) for item in values]

    @staticmethod
    def format_scalar(value: float | None, precision: int) -> str:
        if value is None:
            return "-"
        return f"{float(value):.{precision}f}"

    @staticmethod
    def format_vector(values: list[float], precision: int) -> str:
        joined = ", ".join(f"{float(item):.{precision}f}" for item in values)
        return f"[{joined}]"


class FunctionParser:
    """Safe symbolic parser for f(x)."""

    _allowed_functions = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
    }

    @classmethod
    def parse(cls, expression: str) -> tuple[sp.Symbol, sp.Expr]:
        x_symbol = sp.Symbol("x")
        locals_dict = {"x": x_symbol, **cls._allowed_functions}
        try:
            parsed = sp.sympify(expression, locals=locals_dict, evaluate=True)
        except (sp.SympifyError, TypeError) as exc:
            raise ValueError("Invalid function expression.") from exc

        if parsed.free_symbols - {x_symbol}:
            raise ValueError("Only variable x is allowed in function input.")

        invalid = [
            node
            for node in parsed.atoms(sp.Function)
            if node.func.__name__.lower() not in cls._allowed_functions
        ]
        if invalid:
            raise ValueError("Function contains unsupported symbolic functions.")

        return x_symbol, parsed


def ensure_finite(*values: float) -> bool:
    """Returns True if all values are finite."""
    return all(math.isfinite(float(value)) for value in values)


def is_diagonally_dominant(matrix: np.ndarray) -> bool:
    """Checks strict diagonal dominance."""
    diagonal = np.abs(np.diag(matrix))
    off_diagonal = np.sum(np.abs(matrix), axis=1) - diagonal
    return bool(np.all(diagonal > off_diagonal))
