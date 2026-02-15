import numpy as np

from .models import IterationRow, MethodResult


def validate_system(matrix, constants):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix should be square.")
    if constants.shape != (matrix.shape[0],):
        raise ValueError("Constant vector length should match matrix size.")
    if np.any(np.isclose(np.diag(matrix), 0.0)):
        raise ValueError("Diagonal entries cannot be zero.")


def rounded_vector(values, precision):
    return [round(float(v), precision) for v in values]


class GaussJacobi:
    def __init__(self, matrix, constants, iterations, precision):
        self.matrix = np.array(matrix, dtype=float)
        self.constants = np.array(constants, dtype=float)
        self.iterations = int(iterations)
        self.precision = int(precision)
        validate_system(self.matrix, self.constants)

    def solve(self):
        n = self.matrix.shape[0]
        x_old = np.zeros(n, dtype=float)
        rows = []

        for step in range(1, self.iterations + 1):
            x_new = np.zeros(n, dtype=float)
            for i in range(n):
                sigma = np.dot(self.matrix[i, :], x_old) - self.matrix[i, i] * x_old[i]
                x_new[i] = (self.constants[i] - sigma) / self.matrix[i, i]

            error = float(np.max(np.abs(x_new - x_old)))
            rows.append(IterationRow(step, rounded_vector(x_new, self.precision), round(error, self.precision)))
            x_old = x_new

            if error < 1e-10:
                return MethodResult("Gauss-Jacobi", True, "Values settled with x(0)=0.", rounded_vector(x_new, self.precision), rows)

        return MethodResult("Gauss-Jacobi", False, "Iteration limit reached.", rounded_vector(x_old, self.precision), rows)


class GaussSeidel:
    def __init__(self, matrix, constants, iterations, precision):
        self.matrix = np.array(matrix, dtype=float)
        self.constants = np.array(constants, dtype=float)
        self.iterations = int(iterations)
        self.precision = int(precision)
        validate_system(self.matrix, self.constants)

    def solve(self):
        n = self.matrix.shape[0]
        x = np.zeros(n, dtype=float)
        rows = []

        for step in range(1, self.iterations + 1):
            previous = x.copy()
            for i in range(n):
                left = np.dot(self.matrix[i, :i], x[:i])
                right = np.dot(self.matrix[i, i + 1 :], previous[i + 1 :])
                x[i] = (self.constants[i] - left - right) / self.matrix[i, i]

            error = float(np.max(np.abs(x - previous)))
            rows.append(IterationRow(step, rounded_vector(x, self.precision), round(error, self.precision)))

            if error < 1e-10:
                return MethodResult("Gauss-Seidel", True, "Values settled with x(0)=0.", rounded_vector(x, self.precision), rows)

        return MethodResult("Gauss-Seidel", False, "Iteration limit reached.", rounded_vector(x, self.precision), rows)
