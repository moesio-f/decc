"""Mutation operators.
"""
from __future__ import annotations

import numpy as np


def rand_individual_pertubation(population: np.ndarray,
                                F: float,
                                seed: int,
                                n: int = None) -> np.ndarray:
    """Generate a mutated set of individuals based on the current
    population. The idea is to sample three individuals (a, b and c)
    from the population and use them to create a mutated individual
    t = a + F * (b - c).

    This implementation follows the definition in:
    https://doi.org/10.1007/11539117_147

    Args:
        population (np.ndarray): float array with shape (n, n_dims).
        F (float): scaling factor in [0, 2].
        seed (int): random seed.
        n (int, optional): number of mutated vectors. 
            Defaults to population size.

    Returns:
        array of mutated solutions with shape (n_mutated, n_dims).
    """
    rng = np.random.default_rng(seed)

    if n is None:
        n = population.shape[0]

    # Generating all parameters for the
    #   perturbation.
    # The idea is to generate a random matrix
    #   with size (n, pop_size) and obtain the indices
    #   that would yield the sorted array.
    # From those indices, we can select only the first 3
    #   and obtain the actual individuals to be used
    #   as parameters for the mutation.
    random_matrix = rng.random(size=(n, population.shape[0]))
    random_indices = random_matrix.argsort(axis=1)
    parameters_indices = random_indices[:,:3]
    parameters = population[parameters_indices]

    # Extracting the values of each parameter
    #   in 3 arrays of shape (n, )
    a = parameters[:, 0]
    b = parameters[:, 1]
    c = parameters[:, 2]

    # Calculating the mutated individuals
    #   based on the parameters and F
    mutated_individuals = a + F * (b - c)

    return mutated_individuals
