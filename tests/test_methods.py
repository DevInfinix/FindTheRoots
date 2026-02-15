"""Unit tests for numerical method engine."""

from __future__ import annotations

import unittest

import numpy as np

from numerical_methods import GaussJacobi, GaussSeidel, NewtonRaphson, RegulaFalsi


class RootMethodTests(unittest.TestCase):
    def test_newton_raphson_converges(self) -> None:
        solver = NewtonRaphson(
            function_expression="x**2 - 2",
            initial_guess=1.5,
            max_iterations=25,
            tolerance=1e-10,
            precision=10,
        )
        result = solver.solve()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(float(result.final_estimate), np.sqrt(2), places=8)

    def test_newton_derivative_zero_detected(self) -> None:
        solver = NewtonRaphson(
            function_expression="x**3",
            initial_guess=0.0,
            max_iterations=10,
            tolerance=1e-10,
            precision=8,
        )
        result = solver.solve()
        self.assertFalse(result.converged)
        self.assertTrue(result.diverged)
        self.assertIn("derivative", result.message.lower())

    def test_regula_falsi_converges(self) -> None:
        solver = RegulaFalsi(
            function_expression="x**3 - x - 2",
            lower_bound=1,
            upper_bound=2,
            max_iterations=50,
            tolerance=1e-10,
            precision=10,
        )
        result = solver.solve()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(float(result.final_estimate), 1.5213797068, places=7)

    def test_regula_falsi_invalid_interval_raises(self) -> None:
        with self.assertRaises(ValueError):
            RegulaFalsi(
                function_expression="x**2 + 1",
                lower_bound=-1,
                upper_bound=1,
            )


class LinearMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.array(
            [
                [10.0, 1.0, 1.0],
                [2.0, 10.0, 1.0],
                [2.0, 2.0, 10.0],
            ]
        )
        self.constants = np.array([12.0, 13.0, 14.0])
        self.initial_guess = np.array([0.0, 0.0, 0.0])

    def test_gauss_jacobi_converges(self) -> None:
        solver = GaussJacobi(
            matrix=self.matrix,
            constants=self.constants,
            initial_guess=self.initial_guess,
            max_iterations=100,
            tolerance=1e-9,
            precision=10,
        )
        result = solver.solve()
        self.assertTrue(result.converged)
        self.assertFalse(result.diverged)
        expected = np.linalg.solve(self.matrix, self.constants)
        self.assertTrue(np.allclose(result.final_estimate, expected, atol=1e-6))

    def test_gauss_seidel_converges(self) -> None:
        solver = GaussSeidel(
            matrix=self.matrix,
            constants=self.constants,
            initial_guess=self.initial_guess,
            max_iterations=100,
            tolerance=1e-10,
            precision=10,
        )
        result = solver.solve()
        self.assertTrue(result.converged)
        self.assertFalse(result.diverged)
        expected = np.linalg.solve(self.matrix, self.constants)
        self.assertTrue(np.allclose(result.final_estimate, expected, atol=1e-7))

    def test_zero_diagonal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GaussJacobi(
                matrix=np.array([[0.0, 1.0], [1.0, 2.0]]),
                constants=np.array([1.0, 2.0]),
                initial_guess=np.array([0.0, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()
