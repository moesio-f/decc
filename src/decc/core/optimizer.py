"""Optimizer interface definition.
"""
from __future__ import annotations

import logging
import time
import numpy as np
from abc import ABC, abstractmethod

from .problem import Problem

logger = logging.getLogger(__name__)


class Optimizer(ABC):
    def __init__(self,
                 problem: Problem,
                 seed: int) -> None:
        """Class constructor, random seed must be
        passed to ensure reproducibility.

        Args:
            problem (Problem): problem to solve (minimize).
            seed (int): random seed to be used
                by the optimizer.
        """
        assert problem is not None, 'Problem can\'t be None.'
        assert isinstance(problem, Problem), 'Must be an instance of Problem'
        assert seed is not None, 'Seed can\'t be None.'
        assert seed >= 0, 'Seed must be nonnegative.'
        self.problem = problem
        self.seed = seed

    def optimize(self, *args, **kwargs) -> dict:
        start = time.perf_counter()
        logger.info('Started optimization process...')
        fitness, solution, extras = self._optimize(*args, **kwargs)
        duration = time.perf_counter() - start

        if extras is None:
            extras = dict()

        return dict(duration=duration,
                    best_fitness=fitness,
                    best_solution=solution,
                    **extras)

    @abstractmethod
    def _optimize(self, *args, **kwargs) -> tuple[np.ndarray,
                                                  np.ndarray,
                                                  dict | None]:
        """Runs the optimization process and returns
        the best fitness, best solution and possibly more
        information.

        Returns:
            (best_fitness, best_solution, extras)
        """

    @abstractmethod
    def parameters(self) -> dict:
        """Optimizer parameters.

        Returns:
            dictionary with the optimizer
                parameters.
        """

    @abstractmethod
    def name(self) -> str:
        """Optimizer name.

        Returns:
            optimizer name.
        """

    def __str__(self) -> str:
        return self.name()
