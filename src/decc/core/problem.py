"""Problem interface definition.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


class Problem:
    def __init__(self,
                 objective: Callable[[np.ndarray], np.ndarray],
                 dims: int,
                 lower_bound: float | ArrayLike = -np.inf,
                 upper_bound: float | ArrayLike = np.inf) -> None:
        """Problem constructor. Receives the objective function,
        number of dimensions, lower and upper bounds.

        Args:
            objective (Callable[[np.ndarray], np.ndarray]): objective
                function. The function should work on inputs with
                shape (n, n_dims) and produce output with 
                shape (n,).
            dims (int): number of dimensions of the objective.
            lower_bound (float | ArrayLike): lower bound for the search
                space. Defaults to minus infinity.
            upper_bound (float | ArrayLike): upper bound for the search
                space. Defaults to plus infinity.
        """
        self._fn = objective
        self._d = dims
        self._l = np.broadcast_to(np.array(lower_bound,
                                           dtype=np.float32),
                                  (self._d,))
        self._u = np.broadcast_to(np.array(upper_bound,
                                           dtype=np.float32),
                                  (self._d,))

    @property
    def fn(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._fn

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._l, self._u

    @property
    def dims(self) -> int:
        return self._d
