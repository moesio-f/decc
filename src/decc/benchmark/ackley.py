"""Ackley benchmark function.
"""
from __future__ import annotations

import numpy as np

from decc.core import Problem


class Ackley:
    def __init__(self, 
                 dims: int,
                 lower_bound: float = -32.768,
                 upper_bound: float = 32.768,
                 a: float = 20,
                 b: float = 0.2,
                 c: float = 2 * np.pi):
        self.dims = dims
        self.lower = lower_bound
        self.upper = upper_bound
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: np.ndarray) -> np.ndarray:
        d = self.dims
        sum1 = np.sum(x * x, axis=-1)
        sum2 = np.sum(np.cos(self.c * x), axis=-1)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
        term2 = np.exp(sum2 / d)
        result = term1 - term2 + self.a + np.e

        return result
    
    def as_problem(self) -> Problem:
        return Problem(self, 
                       self.dims,
                       self.lower,
                       self.upper)
