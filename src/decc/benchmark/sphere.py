"""Ackley benchmark function.
"""
from __future__ import annotations

import numpy as np

from decc.core import Problem


class Sphere:
    def __init__(self,
                 dims: int,
                 lower_bound: float = -100.0,
                 upper_bound: float = 100.0) -> None:
        self.dims = dims
        self.lower = lower_bound
        self.upper = upper_bound

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x ** 2, axis=-1)

    def as_problem(self) -> Problem:
        return Problem(self,
                       self.dims,
                       self.lower,
                       self.upper)
