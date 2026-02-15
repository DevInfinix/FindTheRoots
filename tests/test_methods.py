import unittest

import numpy as np

from numerical_methods import GaussJacobi, GaussSeidel, NewtonRaphson, RegulaFalsi


class RootMethodTests(unittest.TestCase):
    def test_newton_raphson_converges(self):
        result = NewtonRaphson("x**2 - 2", 1.5, iterations=20, precision=8).solve()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(float(result.final_value), np.sqrt(2), places=6)

    def test_regula_falsi_converges(self):
        result = RegulaFalsi("x**3 - x - 2", 1.0, 2.0, iterations=30, precision=8).solve()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(float(result.final_value), 1.5213797, places=5)

    def test_regula_falsi_checks_interval(self):
        with self.assertRaises(ValueError):
            RegulaFalsi("x**2 + 1", -1.0, 1.0, iterations=10, precision=6)


class LinearMethodTests(unittest.TestCase):
    def setUp(self):
        self.matrix = [[10.0, -1.0, 2.0, 0.0], [-1.0, 11.0, -1.0, 3.0], [2.0, -1.0, 10.0, -1.0], [0.0, 3.0, -1.0, 8.0]]
        self.constants = [6.0, 25.0, -11.0, 15.0]

    def test_gauss_jacobi_converges(self):
        result = GaussJacobi(self.matrix, self.constants, iterations=60, precision=7).solve()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.final_value[0], 1.0, places=4)
        self.assertAlmostEqual(result.final_value[1], 2.0, places=4)
        self.assertAlmostEqual(result.final_value[2], -1.0, places=4)
        self.assertAlmostEqual(result.final_value[3], 1.0, places=4)

    def test_gauss_seidel_converges(self):
        result = GaussSeidel(self.matrix, self.constants, iterations=30, precision=7).solve()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.final_value[0], 1.0, places=5)
        self.assertAlmostEqual(result.final_value[1], 2.0, places=5)
        self.assertAlmostEqual(result.final_value[2], -1.0, places=5)
        self.assertAlmostEqual(result.final_value[3], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
