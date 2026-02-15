class IterationRow:
    def __init__(self, iteration, value, error=None):
        self.iteration = iteration
        self.value = value
        self.error = error


class MethodResult:
    def __init__(self, method_name, converged, message, final_value, rows):
        self.method_name = method_name
        self.converged = converged
        self.message = message
        self.final_value = final_value
        self.rows = rows
