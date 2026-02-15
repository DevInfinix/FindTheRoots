"""Public API for numerical methods package."""

from .linear_methods import GaussJacobi, GaussSeidel
from .models import IterationRecord, MethodResult
from .root_methods import NewtonRaphson, RegulaFalsi
from .utils import PrecisionFormatter

__all__ = [
    "GaussJacobi",
    "GaussSeidel",
    "IterationRecord",
    "MethodResult",
    "NewtonRaphson",
    "PrecisionFormatter",
    "RegulaFalsi",
]
