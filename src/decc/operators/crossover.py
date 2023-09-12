"""Crossover operators.
"""
from __future__ import annotations

import numpy as np


def dim_prob_crossover(population: np.ndarray,
                       mutated_population: np.ndarray,
                       CR: float,
                       seed: int) -> np.ndarray:
    """Simple probability crossover that creates new individuals 
    from the current individuals and a mutated population.

    This implementation follows the definition in:
    https://doi.org/10.1007/11539117_147 

    Args:
        population (np.ndarray): _description_
        mutated_population (np.ndarray): _description_
        CR (float): _description_
        seed (int): _description_

    Returns:
        np.ndarray: _description_
    """
    rng = np.random.default_rng(seed)

    # Obtaining the number of individuals
    #   and the number of dimensions
    n, d = population.shape

    # Obtaining the indices of each dimension
    dim_indices = np.broadcast_to(np.arange(0, d),
                                  (n, d))

    # Obtaining the probabilities
    probabilities = rng.random(size=(n, d))

    # Obtaining the random indices
    indices = rng.integers(0, d, size=(n, d))

    # Creating a mask to take the values from the
    #   mutated population
    take_from_mutated = np.logical_or(probabilities <= CR,
                                      indices == dim_indices)

    # Obtaining the new individuals
    return np.where(take_from_mutated,
                    mutated_population,
                    population)
