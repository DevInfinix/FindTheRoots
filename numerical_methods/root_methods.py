import math

import sympy as sp

from .models import IterationRow, MethodResult


class NewtonRaphson:
    def __init__(self, function_expression, initial_guess, iterations, precision):
        self.initial_guess = float(initial_guess)
        self.iterations = int(iterations)
        self.precision = int(precision)
        x = sp.Symbol("x")
        expr = sp.sympify(function_expression)
        derivative = sp.diff(expr, x)
        self.f = sp.lambdify(x, expr, modules="numpy")
        self.df = sp.lambdify(x, derivative, modules="numpy")

    def solve(self):
        rows = []
        x_value = self.initial_guess

        for step in range(1, self.iterations + 1):
            f_value = float(self.f(x_value))
            df_value = float(self.df(x_value))
            if abs(df_value) < 1e-12:
                return MethodResult(
                    "Newton-Raphson",
                    False,
                    "Stopped because derivative became zero.",
                    round(x_value, self.precision),
                    rows,
                )

            next_value = x_value - (f_value / df_value)
            if not math.isfinite(next_value):
                return MethodResult(
                    "Newton-Raphson",
                    False,
                    "Stopped because value became non-finite.",
                    None,
                    rows,
                )

            error = abs(next_value - x_value)
            rows.append(IterationRow(step, round(next_value, self.precision), round(error, self.precision)))
            x_value = next_value

            if abs(float(self.f(x_value))) < 1e-10:
                return MethodResult(
                    "Newton-Raphson",
                    True,
                    "Root settled.",
                    round(x_value, self.precision),
                    rows,
                )

        return MethodResult(
            "Newton-Raphson",
            False,
            "Iteration limit reached.",
            round(x_value, self.precision),
            rows,
        )


class RegulaFalsi:
    def __init__(self, function_expression, lower_bound, upper_bound, iterations, precision):
        self.a = float(lower_bound)
        self.b = float(upper_bound)
        self.iterations = int(iterations)
        self.precision = int(precision)
        x = sp.Symbol("x")
        expr = sp.sympify(function_expression)
        self.f = sp.lambdify(x, expr, modules="numpy")

        if self.a >= self.b:
            raise ValueError("Lower bound should be smaller than upper bound.")
        fa = float(self.f(self.a))
        fb = float(self.f(self.b))
        if fa * fb > 0:
            raise ValueError("Choose an interval where f(a) and f(b) have opposite signs.")

    def solve(self):
        rows = []
        a = self.a
        b = self.b
        fa = float(self.f(a))
        fb = float(self.f(b))
        current = a

        for step in range(1, self.iterations + 1):
            denominator = fb - fa
            if abs(denominator) < 1e-12:
                return MethodResult(
                    "Regula Falsi",
                    False,
                    "Stopped because denominator became too small.",
                    round(current, self.precision),
                    rows,
                )

            current = (a * fb - b * fa) / denominator
            fc = float(self.f(current))
            rows.append(IterationRow(step, round(current, self.precision), round(abs(fc), self.precision)))

            if abs(fc) < 1e-10:
                return MethodResult(
                    "Regula Falsi",
                    True,
                    "Root settled.",
                    round(current, self.precision),
                    rows,
                )

            if fa * fc < 0:
                b = current
                fb = fc
            else:
                a = current
                fa = fc

        return MethodResult(
            "Regula Falsi",
            False,
            "Iteration limit reached.",
            round(current, self.precision),
            rows,
        )
