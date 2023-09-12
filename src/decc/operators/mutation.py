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
    values = []
    for _ in range(n):
        values.append(rng.choice(population,
                                 size=(3,),
                                 replace=False,
                                 axis=0))

    # Converting the values to an array of shape
    #   (n, 3)
    parameters = np.stack(values)

    # Extracting the values of each parameter
    #   in 3 arrays of shape (n, )
    a = parameters[:, 0]
    b = parameters[:, 1]
    c = parameters[:, 2]

    # Calculating the mutated individuals
    #   based on the parameters and F
    mutated_individuals = a + F * (b - c)

    return mutated_individuals
