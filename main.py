"""Single-file Numerical Methods Visualizer desktop application."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Union

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import sympy as sp



"""Utility helpers for precision and validation."""





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


"""Datamodels shared by numerical method implementations."""




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


"""Abstract class hierarchy for numerical methods."""






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


"""Root-finding strategy implementations."""





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


"""Linear-system iterative method implementations."""






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


"""Strategy factory for solver instantiation."""






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


"""UI theme constants."""


PALETTE = {
    "bg": "#0A0E14",
    "surface": "#141A22",
    "card": "#1B2430",
    "accent": "#2AA9FF",
    "accent_hover": "#62C2FF",
    "text_primary": "#E6EDF7",
    "text_secondary": "#9EB0C7",
    "success": "#2ECC71",
    "danger": "#FF5C7A",
}

FONTS = {
    "title": ("Segoe UI Semibold", 36),
    "subtitle": ("Segoe UI", 16),
    "heading": ("Segoe UI Semibold", 24),
    "body": ("Segoe UI", 14),
    "mono": ("Cascadia Code", 12),
}


"""Base frame class used across all application screens."""




class AppFrame(ctk.CTkFrame):
    """Base frame with lifecycle hooks."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.controller = controller

    def on_show(self) -> None:
        """Called whenever the frame is raised."""
        return


"""Main UI frame implementations."""







@dataclass(slots=True)
class MethodCardSpec:
    """Metadata used to render each selection card."""

    key: str
    name: str
    description: str
    formula: str


class LandingFrame(AppFrame):
    """Landing and onboarding screen."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        card = ctk.CTkFrame(
            self,
            fg_color=PALETTE["surface"],
            corner_radius=18,
            border_width=1,
            border_color="#223042",
        )
        card.grid(row=0, column=0, padx=48, pady=48, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        self._full_title = "Numerical Methods Visualizer"
        self._title_index = 0

        self.title_label = ctk.CTkLabel(
            card,
            text="",
            font=FONTS["title"],
            text_color=PALETTE["text_primary"],
        )
        self.title_label.grid(row=0, column=0, padx=24, pady=(28, 8), sticky="n")

        subtitle_text = (
            "Explore root-finding and linear-system iterative solvers with precise, "
            "iteration-by-iteration numerical feedback."
        )
        subtitle = ctk.CTkLabel(
            card,
            text=subtitle_text,
            justify="center",
            wraplength=860,
            font=FONTS["subtitle"],
            text_color=PALETTE["text_secondary"],
        )
        subtitle.grid(row=1, column=0, padx=28, pady=(0, 20))

        info_card = ctk.CTkFrame(card, fg_color=PALETTE["card"], corner_radius=14)
        info_card.grid(row=2, column=0, padx=28, pady=8, sticky="ew")
        info_card.grid_columnconfigure((0, 1, 2), weight=1)

        self._build_info_column(
            info_card,
            0,
            "Root-Finding",
            "Newton-Raphson and Regula Falsi locate x where f(x)=0.",
        )
        self._build_info_column(
            info_card,
            1,
            "Iterative Solvers",
            "Gauss-Jacobi and Gauss-Seidel solve Ax=b using repeated updates.",
        )
        self._build_info_column(
            info_card,
            2,
            "Convergence",
            "Track per-iteration error to verify stability and stopping criteria.",
        )

        cta = ctk.CTkButton(
            card,
            text="Get Started",
            width=220,
            height=44,
            corner_radius=10,
            font=FONTS["subtitle"],
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            text_color="#04121E",
            command=lambda: controller.show_frame("SelectionFrame"),
        )
        cta.grid(row=3, column=0, pady=(24, 30))

    def _build_info_column(self, parent: ctk.CTkFrame, column: int, title: str, body: str) -> None:
        col_frame = ctk.CTkFrame(parent, fg_color="transparent")
        col_frame.grid(row=0, column=column, padx=14, pady=18, sticky="nsew")

        ctk.CTkLabel(
            col_frame,
            text=title,
            font=FONTS["heading"],
            text_color=PALETTE["text_primary"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            col_frame,
            text=body,
            wraplength=250,
            justify="left",
            font=FONTS["body"],
            text_color=PALETTE["text_secondary"],
        ).pack(anchor="w", pady=(6, 0))

    def on_show(self) -> None:
        self._title_index = 0
        self.title_label.configure(text="")
        self._animate_title()

    def _animate_title(self) -> None:
        if self._title_index <= len(self._full_title):
            self.title_label.configure(text=self._full_title[: self._title_index])
            self._title_index += 1
            self.after(22, self._animate_title)


class SelectionFrame(AppFrame):
    """Card-based method selection screen."""

    CARD_COLOR = "#1A212D"
    CARD_HOVER_COLOR = "#233044"

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        panel = ctk.CTkFrame(
            self,
            fg_color=PALETTE["surface"],
            corner_radius=16,
            border_width=1,
            border_color="#26364C",
        )
        panel.grid(row=0, column=0, padx=42, pady=42, sticky="nsew")
        panel.grid_columnconfigure((0, 1), weight=1)
        panel.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            panel,
            text="Choose a Numerical Method",
            font=FONTS["title"],
            text_color=PALETTE["text_primary"],
        ).grid(row=0, column=0, columnspan=2, pady=(26, 6))
        ctk.CTkLabel(
            panel,
            text="Select a strategy to configure inputs and run iteration analysis.",
            font=FONTS["subtitle"],
            text_color=PALETTE["text_secondary"],
        ).grid(row=1, column=0, columnspan=2, pady=(0, 20))

        self.method_specs = [
            MethodCardSpec(
                key="newton_raphson",
                name="Newton-Raphson",
                description="Fast tangent-based root refinement from one initial guess.",
                formula="x(n+1) = x(n) - f(x(n)) / f'(x(n))",
            ),
            MethodCardSpec(
                key="regula_falsi",
                name="Regula Falsi",
                description="Bracketed false-position method using interval sign changes.",
                formula="c = (a f(b) - b f(a)) / (f(b) - f(a))",
            ),
            MethodCardSpec(
                key="gauss_jacobi",
                name="Gauss-Jacobi",
                description="Parallel-friendly iterative solver using prior vector values.",
                formula="x_i(k+1) = (b_i - Σ(j!=i) a_ij x_j(k)) / a_ii",
            ),
            MethodCardSpec(
                key="gauss_seidel",
                name="Gauss-Seidel",
                description="Sequential iterative solver with immediate in-step updates.",
                formula="x_i(k+1) = (b_i - Σ(j<i) a_ij x_j(k+1) - Σ(j>i) a_ij x_j(k)) / a_ii",
            ),
        ]

        cards_frame = ctk.CTkFrame(panel, fg_color="transparent")
        cards_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=24, pady=10)
        cards_frame.grid_columnconfigure((0, 1), weight=1)
        cards_frame.grid_rowconfigure((0, 1), weight=1)

        for index, spec in enumerate(self.method_specs):
            row = index // 2
            col = index % 2
            card = self._build_method_card(cards_frame, spec)
            card.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")

        footer = ctk.CTkFrame(panel, fg_color="transparent")
        footer.grid(row=3, column=0, columnspan=2, pady=(6, 20))
        ctk.CTkButton(
            footer,
            text="Back",
            width=120,
            fg_color="#29364A",
            hover_color="#374A66",
            command=lambda: controller.show_frame("LandingFrame"),
        ).pack()

    def _build_method_card(self, parent: ctk.CTkFrame, spec: MethodCardSpec) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent,
            fg_color=self.CARD_COLOR,
            corner_radius=14,
            border_width=1,
            border_color="#2A3A50",
        )

        title = ctk.CTkLabel(card, text=spec.name, font=FONTS["heading"], text_color=PALETTE["text_primary"])
        title.pack(anchor="w", padx=16, pady=(14, 6))

        description = ctk.CTkLabel(
            card,
            text=spec.description,
            wraplength=420,
            justify="left",
            font=FONTS["body"],
            text_color=PALETTE["text_secondary"],
        )
        description.pack(anchor="w", padx=16)

        formula = ctk.CTkLabel(
            card,
            text=spec.formula,
            wraplength=420,
            justify="left",
            font=FONTS["mono"],
            text_color="#8DD5FF",
        )
        formula.pack(anchor="w", padx=16, pady=(10, 12))

        select_button = ctk.CTkButton(
            card,
            text="Select",
            width=100,
            height=34,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            text_color="#03111D",
            command=lambda selected=spec.key: self._select_method(selected),
        )
        select_button.pack(anchor="w", padx=16, pady=(0, 14))

        self._bind_card_hover(card, [title, description, formula, select_button])
        return card

    def _bind_card_hover(self, card: ctk.CTkFrame, children: list[ctk.CTkBaseClass]) -> None:
        targets = [card, *children]
        for target in targets:
            target.bind("<Enter>", lambda _event, c=card: c.configure(fg_color=self.CARD_HOVER_COLOR))
            target.bind("<Leave>", lambda _event, c=card: c.configure(fg_color=self.CARD_COLOR))

    def _select_method(self, method_key: str) -> None:
        self.controller.selected_method = method_key
        self.controller.show_frame("InputFrame")


class InputFrame(AppFrame):
    """Dynamic method-specific input form screen."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])
        self.method_names = {
            "newton_raphson": "Newton-Raphson",
            "regula_falsi": "Regula Falsi",
            "gauss_jacobi": "Gauss-Jacobi",
            "gauss_seidel": "Gauss-Seidel",
        }

        self.matrix_size = ctk.StringVar(value="3")
        self.function_entry: ctk.CTkEntry | None = None
        self.initial_guess_entry: ctk.CTkEntry | None = None
        self.lower_bound_entry: ctk.CTkEntry | None = None
        self.upper_bound_entry: ctk.CTkEntry | None = None
        self.iterations_entry: ctk.CTkEntry | None = None
        self.precision_entry: ctk.CTkEntry | None = None
        self.tolerance_entry: ctk.CTkEntry | None = None
        self.matrix_entries: list[list[ctk.CTkEntry]] = []
        self.constants_entries: list[ctk.CTkEntry] = []
        self.initial_vector_entries: list[ctk.CTkEntry] = []

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.container = ctk.CTkFrame(
            self,
            fg_color=PALETTE["surface"],
            corner_radius=16,
            border_width=1,
            border_color="#26364C",
        )
        self.container.grid(row=0, column=0, padx=38, pady=38, sticky="nsew")
        self.container.grid_rowconfigure(1, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.container,
            text="Configure Method",
            font=FONTS["title"],
            text_color=PALETTE["text_primary"],
        )
        self.title_label.grid(row=0, column=0, pady=(20, 6))

        self.form_scroll = ctk.CTkScrollableFrame(self.container, fg_color=PALETTE["card"], corner_radius=14)
        self.form_scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=(6, 12))
        self.form_scroll.grid_columnconfigure(0, weight=1)

        self.error_label = ctk.CTkLabel(
            self.container,
            text="",
            font=FONTS["body"],
            text_color=PALETTE["danger"],
        )
        self.error_label.grid(row=2, column=0, pady=(0, 6))

        footer = ctk.CTkFrame(self.container, fg_color="transparent")
        footer.grid(row=3, column=0, pady=(0, 20))
        ctk.CTkButton(
            footer,
            text="Back",
            width=110,
            fg_color="#29364A",
            hover_color="#374A66",
            command=lambda: controller.show_frame("SelectionFrame"),
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            footer,
            text="Solve",
            width=150,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            text_color="#04121E",
            command=self._solve_current_method,
        ).pack(side="left", padx=8)

    def on_show(self) -> None:
        method_key = self.controller.selected_method or "newton_raphson"
        method_name = self.method_names.get(method_key, "Method")
        self.title_label.configure(text=f"Configure {method_name}")
        self.error_label.configure(text="")
        self._render_method_form(method_key)

    def _render_method_form(self, method_key: str) -> None:
        for child in self.form_scroll.winfo_children():
            child.destroy()

        self.function_entry = None
        self.initial_guess_entry = None
        self.lower_bound_entry = None
        self.upper_bound_entry = None
        self.matrix_entries = []
        self.constants_entries = []
        self.initial_vector_entries = []

        row = 0
        if method_key in {"newton_raphson", "regula_falsi"}:
            self.function_entry = self._add_entry_row(row, "f(x)", "x**3 - x - 2")
            row += 1

            if method_key == "newton_raphson":
                self.initial_guess_entry = self._add_entry_row(row, "Initial Guess", "1.5")
            else:
                self.lower_bound_entry = self._add_entry_row(row, "Lower Bound (a)", "1")
                row += 1
                self.upper_bound_entry = self._add_entry_row(row, "Upper Bound (b)", "2")
            row += 1
        else:
            self._add_matrix_size_selector(row)
            row += 1
            self._build_matrix_grid(size=int(self.matrix_size.get()), row=row)
            row += 1

        self.iterations_entry = self._add_entry_row(row, "Max Iterations", "50")
        row += 1
        self.precision_entry = self._add_entry_row(row, "Decimal Precision", "6")
        row += 1
        self.tolerance_entry = self._add_entry_row(row, "Tolerance", "1e-8")

    def _add_entry_row(self, row: int, label: str, placeholder: str) -> ctk.CTkEntry:
        wrapper = ctk.CTkFrame(self.form_scroll, fg_color="transparent")
        wrapper.grid(row=row, column=0, sticky="ew", padx=16, pady=8)
        wrapper.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(wrapper, text=label, width=180, anchor="w", font=FONTS["body"]).grid(row=0, column=0, padx=(0, 8))
        entry = ctk.CTkEntry(wrapper, placeholder_text=placeholder, fg_color="#111824", border_color="#32445E")
        entry.grid(row=0, column=1, sticky="ew")
        return entry

    def _add_matrix_size_selector(self, row: int) -> None:
        wrapper = ctk.CTkFrame(self.form_scroll, fg_color="transparent")
        wrapper.grid(row=row, column=0, sticky="ew", padx=16, pady=8)
        wrapper.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(wrapper, text="Matrix Size", width=180, anchor="w", font=FONTS["body"]).grid(row=0, column=0, padx=(0, 8))
        selector = ctk.CTkOptionMenu(
            wrapper,
            variable=self.matrix_size,
            values=["2", "3", "4", "5", "6"],
            fg_color="#2A3A50",
            button_color="#324760",
            button_hover_color="#3E5676",
            command=lambda value: self._build_matrix_grid(size=int(value), row=row + 1),
        )
        selector.grid(row=0, column=1, sticky="w")

    def _build_matrix_grid(self, size: int, row: int) -> None:
        current = self.form_scroll.grid_slaves(row=row, column=0)
        for widget in current:
            widget.destroy()

        panel = ctk.CTkFrame(self.form_scroll, fg_color="#111824", corner_radius=12)
        panel.grid(row=row, column=0, sticky="ew", padx=16, pady=8)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text="Enter A matrix, b constants, and initial guess vector x(0)",
            font=FONTS["body"],
            text_color=PALETTE["text_secondary"],
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 8))

        grid_frame = ctk.CTkFrame(panel, fg_color="transparent")
        grid_frame.grid(row=1, column=0, padx=14, pady=(0, 12), sticky="w")

        self.matrix_entries = []
        self.constants_entries = []
        self.initial_vector_entries = []

        for i in range(size):
            matrix_row_entries: list[ctk.CTkEntry] = []
            for j in range(size):
                entry = ctk.CTkEntry(grid_frame, width=68, placeholder_text="0")
                entry.grid(row=i, column=j, padx=4, pady=4)
                entry.insert(0, "0")
                matrix_row_entries.append(entry)
            ctk.CTkLabel(grid_frame, text="|", font=FONTS["heading"]).grid(row=i, column=size, padx=8)
            b_entry = ctk.CTkEntry(grid_frame, width=68, placeholder_text="0")
            b_entry.grid(row=i, column=size + 1, padx=4, pady=4)
            b_entry.insert(0, "0")
            self.constants_entries.append(b_entry)

            ctk.CTkLabel(grid_frame, text="x0", font=FONTS["body"]).grid(row=i, column=size + 2, padx=(12, 4))
            x0_entry = ctk.CTkEntry(grid_frame, width=68, placeholder_text="0")
            x0_entry.grid(row=i, column=size + 3, padx=4, pady=4)
            x0_entry.insert(0, "0")
            self.initial_vector_entries.append(x0_entry)

            self.matrix_entries.append(matrix_row_entries)

    def _parse_float(self, entry: ctk.CTkEntry, field_name: str) -> float:
        value = entry.get().strip()
        if not value:
            raise ValueError(f"{field_name} is required.")
        return float(value)

    def _parse_int(self, entry: ctk.CTkEntry, field_name: str, minimum: int = 1) -> int:
        value = entry.get().strip()
        if not value:
            raise ValueError(f"{field_name} is required.")
        parsed = int(value)
        if parsed < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}.")
        return parsed

    def _collect_payload(self, method_key: str) -> dict[str, Any]:
        max_iterations = self._parse_int(self.iterations_entry, "Max Iterations", minimum=1)
        precision = self._parse_int(self.precision_entry, "Decimal Precision", minimum=0)
        tolerance = self._parse_float(self.tolerance_entry, "Tolerance")
        if tolerance <= 0:
            raise ValueError("Tolerance must be positive.")

        payload: dict[str, Any] = {
            "max_iterations": max_iterations,
            "precision": precision,
            "tolerance": tolerance,
        }

        if method_key in {"newton_raphson", "regula_falsi"}:
            function_expression = (self.function_entry.get() if self.function_entry else "").strip()
            if not function_expression:
                raise ValueError("Function expression is required.")
            payload["function_expression"] = function_expression

            if method_key == "newton_raphson":
                payload["initial_guess"] = self._parse_float(self.initial_guess_entry, "Initial Guess")
            else:
                payload["lower_bound"] = self._parse_float(self.lower_bound_entry, "Lower Bound")
                payload["upper_bound"] = self._parse_float(self.upper_bound_entry, "Upper Bound")
        else:
            if not self.matrix_entries:
                raise ValueError("Matrix entries are not initialized.")
            matrix = [
                [self._parse_float(entry, f"A[{row + 1},{col + 1}]") for col, entry in enumerate(row_entries)]
                for row, row_entries in enumerate(self.matrix_entries)
            ]
            constants = [self._parse_float(entry, f"b[{idx + 1}]") for idx, entry in enumerate(self.constants_entries)]
            initial_guess = [
                self._parse_float(entry, f"x0[{idx + 1}]") for idx, entry in enumerate(self.initial_vector_entries)
            ]
            payload["matrix"] = matrix
            payload["constants"] = constants
            payload["initial_guess"] = initial_guess

        return payload

    def _solve_current_method(self) -> None:
        method_key = self.controller.selected_method
        if not method_key:
            self.error_label.configure(text="No method selected. Go back and choose a method first.")
            return

        try:
            payload = self._collect_payload(method_key)
            request = SolveRequest(method_key=method_key, data=payload)
            solver = MethodStrategyFactory.create_solver(request)
            result = solver.solve()
        except Exception as exc:
            self.error_label.configure(text=str(exc))
            return

        self.error_label.configure(text="")
        self.controller.last_result = result
        self.controller.last_request = payload
        self.controller.show_frame("ResultFrame")


class ResultFrame(AppFrame):
    """Results view with iteration table and convergence graph."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])
        self.graph_canvas: FigureCanvasTkAgg | None = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.container = ctk.CTkFrame(
            self,
            fg_color=PALETTE["surface"],
            corner_radius=16,
            border_width=1,
            border_color="#26364C",
        )
        self.container.grid(row=0, column=0, padx=38, pady=38, sticky="nsew")
        self.container.grid_rowconfigure(1, weight=1)
        self.container.grid_columnconfigure(0, weight=3)
        self.container.grid_columnconfigure(1, weight=2)

        self.header_label = ctk.CTkLabel(
            self.container,
            text="Computation Results",
            font=FONTS["title"],
            text_color=PALETTE["text_primary"],
        )
        self.header_label.grid(row=0, column=0, columnspan=2, pady=(18, 4))

        self.summary_label = ctk.CTkLabel(
            self.container,
            text="",
            justify="left",
            font=FONTS["subtitle"],
            text_color=PALETTE["text_secondary"],
        )
        self.summary_label.grid(row=0, column=0, columnspan=2, sticky="s", pady=(64, 12))

        table_card = ctk.CTkFrame(self.container, fg_color=PALETTE["card"], corner_radius=14)
        table_card.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(4, 12))
        table_card.grid_rowconfigure(1, weight=1)
        table_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            table_card,
            text="Iteration Table",
            font=FONTS["heading"],
            text_color=PALETTE["text_primary"],
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 6))

        self.table_text = ctk.CTkTextbox(
            table_card,
            fg_color="#111824",
            corner_radius=10,
            font=FONTS["mono"],
            wrap="none",
            text_color=PALETTE["text_primary"],
        )
        self.table_text.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.table_text.configure(state="disabled")

        right_panel = ctk.CTkFrame(self.container, fg_color=PALETTE["card"], corner_radius=14)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(4, 12))
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right_panel,
            text="Convergence Graph",
            font=FONTS["heading"],
            text_color=PALETTE["text_primary"],
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 6))

        self.graph_host = ctk.CTkFrame(right_panel, fg_color="#111824", corner_radius=10)
        self.graph_host.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))

        footer = ctk.CTkFrame(right_panel, fg_color="transparent")
        footer.grid(row=2, column=0, pady=(0, 12))
        ctk.CTkButton(
            footer,
            text="Edit Inputs",
            width=120,
            fg_color="#29364A",
            hover_color="#374A66",
            command=lambda: controller.show_frame("InputFrame"),
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            footer,
            text="New Method",
            width=120,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            text_color="#04121E",
            command=lambda: controller.show_frame("SelectionFrame"),
        ).pack(side="left", padx=6)

    def on_show(self) -> None:
        self._render_result()

    def _render_result(self) -> None:
        result = self.controller.last_result
        request = self.controller.last_request or {}

        if result is None:
            self.summary_label.configure(text="No results available yet.")
            self._clear_table()
            self._clear_graph()
            return

        precision = int(request.get("precision", 6))
        approx = self._format_estimate(result.final_estimate, precision)
        status = "Converged" if result.converged else ("Diverged" if result.diverged else "Not Converged")
        status_color = (
            PALETTE["success"] if result.converged else (PALETTE["danger"] if result.diverged else "#F5B041")
        )
        self.header_label.configure(text=f"{result.method_name} Results")
        warning_text = ""
        if result.warnings:
            warning_text = f"    Warning: {result.warnings[0]}"
        self.summary_label.configure(
            text=f"Status: {status}    Final Approximation: {approx}    Message: {result.message}{warning_text}",
            text_color=status_color,
        )

        self._populate_table(result.iterations, precision)
        self._plot_convergence(result.iterations)

    def _format_estimate(self, estimate: Any, precision: int) -> str:
        if estimate is None:
            return "-"
        if isinstance(estimate, list):
            return PrecisionFormatter.format_vector(estimate, precision)
        return PrecisionFormatter.format_scalar(float(estimate), precision)

    def _clear_table(self) -> None:
        self.table_text.configure(state="normal")
        self.table_text.delete("1.0", "end")
        self.table_text.configure(state="disabled")

    def _populate_table(self, iterations: list[Any], precision: int) -> None:
        self._clear_table()
        lines = [f"{'Iter':<6}{'Estimate':<44}{'Error':<18}{'Residual':<18}"]
        lines.append("-" * 90)

        for record in iterations:
            estimate_text = self._format_estimate(record.estimate, precision)
            error_text = PrecisionFormatter.format_scalar(record.error, precision)
            residual_text = PrecisionFormatter.format_scalar(record.residual, precision)
            lines.append(f"{record.iteration:<6}{estimate_text:<44}{error_text:<18}{residual_text:<18}")

        self.table_text.configure(state="normal")
        self.table_text.insert("1.0", "\n".join(lines))
        self.table_text.configure(state="disabled")

    def _clear_graph(self) -> None:
        if self.graph_canvas is not None:
            self.graph_canvas.get_tk_widget().destroy()
            self.graph_canvas = None

    def _plot_convergence(self, iterations: list[Any]) -> None:
        self._clear_graph()

        error_points = [(record.iteration, record.error) for record in iterations if record.error is not None]
        residual_points = [
            (record.iteration, record.residual) for record in iterations if record.residual is not None
        ]

        max_points = 600
        if len(error_points) > max_points:
            stride = max(1, len(error_points) // max_points)
            error_points = error_points[::stride]
        if len(residual_points) > max_points:
            stride = max(1, len(residual_points) // max_points)
            residual_points = residual_points[::stride]

        figure = Figure(figsize=(4.8, 3.2), dpi=100, facecolor="#111824")
        axis = figure.add_subplot(111)
        axis.set_facecolor("#111824")
        axis.tick_params(colors="#A7BED7", labelsize=8)
        axis.spines["bottom"].set_color("#3F5878")
        axis.spines["top"].set_color("#3F5878")
        axis.spines["left"].set_color("#3F5878")
        axis.spines["right"].set_color("#3F5878")
        axis.set_xlabel("Iteration", color="#A7BED7")
        axis.set_ylabel("Magnitude", color="#A7BED7")

        if error_points:
            axis.plot(
                [item[0] for item in error_points],
                [item[1] for item in error_points],
                color="#40C4FF",
                linewidth=2,
                label="Error",
            )
        if residual_points:
            axis.plot(
                [item[0] for item in residual_points],
                [item[1] for item in residual_points],
                color="#5DE8A0",
                linewidth=2,
                label="Residual",
            )
        if error_points or residual_points:
            axis.legend(facecolor="#111824", edgecolor="#3F5878", labelcolor="#D4E2F2", fontsize=8)
        else:
            axis.text(0.5, 0.5, "No error/residual data", color="#A7BED7", ha="center", va="center")

        figure.tight_layout(pad=1.2)
        self.graph_canvas = FigureCanvasTkAgg(figure, master=self.graph_host)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)


"""Application shell and frame router."""





class App(ctk.CTk):
    """Main desktop application."""

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Numerical Methods Visualizer")
        self.geometry("1220x760")
        self.minsize(980, 620)
        self.configure(fg_color=PALETTE["bg"])

        container = ctk.CTkFrame(self, fg_color=PALETTE["bg"])
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.selected_method: str | None = None
        self.last_result = None
        self.last_request = None

        self.frames: dict[str, ctk.CTkFrame] = {}
        for frame_cls in (LandingFrame, SelectionFrame, InputFrame, ResultFrame):
            frame = frame_cls(container, self)
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[frame_cls.__name__] = frame

        self.show_frame("LandingFrame")

    def show_frame(self, frame_name: str) -> None:
        frame = self.frames[frame_name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()


def main() -> None:
    """Application bootstrap."""

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
