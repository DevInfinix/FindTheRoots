from numerical_methods import GaussJacobi, GaussSeidel, NewtonRaphson, RegulaFalsi


def create_solver(method_key, data):
    if method_key == "newton_raphson":
        return NewtonRaphson(data["function_expression"], data["initial_guess"], data["iterations"], data["precision"])

    if method_key == "regula_falsi":
        return RegulaFalsi(data["function_expression"], data["lower_bound"], data["upper_bound"], data["iterations"], data["precision"])

    if method_key == "gauss_jacobi":
        return GaussJacobi(data["matrix"], data["constants"], data["iterations"], data["precision"])

    if method_key == "gauss_seidel":
        return GaussSeidel(data["matrix"], data["constants"], data["iterations"], data["precision"])

    raise ValueError("Unsupported method.")
