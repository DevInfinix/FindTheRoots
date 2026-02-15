"""Automated tests for numerical methods core engine."""

from __future__ import annotations

import math
import unittest

from numerical_methods import GaussJacobi, GaussSeidel, NewtonRaphson, RegulaFalsi
from numerical_methods.utils import PrecisionFormatter


class TestRootMethods(unittest.TestCase):
    def test_newton_raphson_converges(self) -> None:
        solver = NewtonRaphson(
            function_expression="x**2 - 2",
            initial_guess=1.0,
            max_iterations=40,
            tolerance=1e-10,
            precision=8,
        )
        result = solver.solve()

        self.assertTrue(result.converged)
        self.assertFalse(result.diverged)
        self.assertAlmostEqual(float(result.final_estimate), math.sqrt(2), places=8)
        self.assertGreater(len(result.iterations), 0)

    def test_newton_handles_zero_derivative(self) -> None:
        solver = NewtonRaphson(function_expression="x**3", initial_guess=0.0, max_iterations=5)
        result = solver.solve()

        self.assertFalse(result.converged)
        self.assertTrue(result.diverged)
        self.assertIn("derivative", result.message.lower())

    def test_regula_falsi_interval_validation(self) -> None:
        with self.assertRaises(ValueError):
            RegulaFalsi(function_expression="x**2 + 1", lower_bound=-1.0, upper_bound=1.0, max_iterations=20)

    def test_regula_falsi_converges(self) -> None:
        solver = RegulaFalsi(
            function_expression="x**3 - x - 2",
            lower_bound=1.0,
            upper_bound=2.0,
            max_iterations=100,
            tolerance=1e-8,
        )
        result = solver.solve()

        self.assertTrue(result.converged)
        self.assertFalse(result.diverged)
        self.assertAlmostEqual(float(result.final_estimate), 1.5213797, places=6)


class TestLinearMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix_a = [[10.0, -1.0, 2.0, 0.0], [-1.0, 11.0, -1.0, 3.0], [2.0, -1.0, 10.0, -1.0], [0.0, 3.0, -1.0, 8.0]]
        self.vector_b = [6.0, 25.0, -11.0, 15.0]
        self.initial_guess = [0.0, 0.0, 0.0, 0.0]

    def test_gauss_jacobi_converges(self) -> None:
        solver = GaussJacobi(
            matrix=self.matrix_a,
            constants=self.vector_b,
            initial_guess=self.initial_guess,
            max_iterations=200,
            tolerance=1e-10,
        )
        result = solver.solve()

        self.assertTrue(result.converged)
        self.assertFalse(result.diverged)
        self.assertAlmostEqual(result.final_estimate[0], 1.0, places=6)
        self.assertAlmostEqual(result.final_estimate[1], 2.0, places=6)
        self.assertAlmostEqual(result.final_estimate[2], -1.0, places=6)
        self.assertAlmostEqual(result.final_estimate[3], 1.0, places=6)

    def test_gauss_seidel_converges(self) -> None:
        solver = GaussSeidel(
            matrix=self.matrix_a,
            constants=self.vector_b,
            initial_guess=self.initial_guess,
            max_iterations=200,
            tolerance=1e-10,
        )
        result = solver.solve()

        self.assertTrue(result.converged)
        self.assertFalse(result.diverged)
        self.assertAlmostEqual(result.final_estimate[0], 1.0, places=8)
        self.assertAlmostEqual(result.final_estimate[1], 2.0, places=8)
        self.assertAlmostEqual(result.final_estimate[2], -1.0, places=8)
        self.assertAlmostEqual(result.final_estimate[3], 1.0, places=8)

    def test_non_diagonally_dominant_warns(self) -> None:
        solver = GaussJacobi(
            matrix=[[1.0, 3.0], [2.0, 1.0]],
            constants=[5.0, 5.0],
            initial_guess=[0.0, 0.0],
            max_iterations=5,
        )
        result = solver.solve()

        self.assertGreater(len(result.warnings), 0)


class TestPrecisionFormatter(unittest.TestCase):
    def test_formatting(self) -> None:
        self.assertEqual(PrecisionFormatter.format_scalar(1.234567, 3), "1.235")
        self.assertEqual(PrecisionFormatter.format_vector([1.2, 3.456], 2), "[1.20, 3.46]")
        self.assertEqual(PrecisionFormatter.round_scalar(-1e-12, 6), 0.0)


if __name__ == "__main__":
    unittest.main()
