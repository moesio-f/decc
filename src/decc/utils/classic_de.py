"""Utility for evolving a
population with classic DE's
algorithms.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from decc.operators import crossover, mutation, selection


def de_rand_1_exp(population: np.ndarray,
                  fitness: np.ndarray,
                  F: float,
                  CR: float,
                  fn: Callable[[np.ndarray], np.ndarray],
                  seed: int,
                  max_fn: int | None = None,
                  bounds: tuple | None = None) -> tuple[np.ndarray,
                                                        np.ndarray,
                                                        int]:
    """Applies a variant of DE known as DE/Rand/1/Exp, this implementation
    uses the description provided in: 
    Shi, Yj., Teng, Hf., Li, Zq.
    Cooperative Co-evolutionary Differential Evolution 
        for Function Optimization (2005) 
    https://doi.org/10.1007/11539117_147


    Args:
        population (np.ndarray): initial population to evolve.
        fitness (np.ndarray): initial fitness values for population.
        F (float): mutation factor.
        CR (float): crossover probability.
        fn (Callable[[np.ndarray], np.ndarray]): fitness function.
        seed (int): random seed.
        max_fn (int | None, optional): maximum number of fitness
            evaluations. Defaults to number of individuals in
            population.

    Returns:
        (population, best_fitness, n_evaluations)
    """
    if max_fn is None:
        max_fn = population.shape[0]

    # Auxiliary variables
    n_evaluations = 0

    # Generator
    rng = np.random.default_rng(seed)

    # Evolution loop
    while n_evaluations < max_fn:
        # Mutation
        mutated_pop = mutation.rand_individual_pertubation(
            population,
            F,
            rng.integers(0, 99999),
            bounds=bounds)

        # Crossover
        candidates = crossover.dim_prob_crossover(
            population,
            mutated_pop,
            CR,
            rng.integers(0, 99999))

        # Candidate evaluation
        candidates_fitness = fn(candidates)
        n_evaluations += candidates.shape[0]

        # Update population based on fitness
        population, fitness = selection.best_fitness_selection(
            population,
            candidates,
            fitness,
            candidates_fitness)

    return population, fitness, n_evaluations
