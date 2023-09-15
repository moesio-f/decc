"""Selection operators.
"""
from __future__ import annotations

import numpy as np


def best_fitness_selection(
        population: np.ndarray,
        candidates: np.ndarray,
        population_fitness: np.ndarray,
        candidates_fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select the best individuals between the current
    population and the candidates.

    Args:
        population (np.ndarray): population.
        candidates (np.ndarray): candidates.
        population_fitness (np.ndarray): population fitness.
        candidates_fitness (np.ndarray): candidates fitness.

    Returns:
        (new_population, fitness)
    """

    # Find where the candidates are better than the population
    cond = candidates_fitness <= population_fitness

    # Create the same condition but for a row of the population
    cond_dims = np.repeat(np.expand_dims(cond, axis=-1),
                          candidates.shape[-1],
                          axis=-1)

    # Select the individuals with best fitness
    population = np.where(cond_dims,
                          candidates,
                          population)

    # Select the best fitness for those individuals
    fitness = np.where(cond,
                       candidates_fitness,
                       population_fitness)

    return population, fitness
