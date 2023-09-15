"""Mutation operators.
"""
from __future__ import annotations

import numpy as np


def rand_individual_pertubation(population: np.ndarray,
                                F: float | np.ndarray,
                                seed: int,
                                n: int = None,
                                bounds: tuple = None) -> np.ndarray:
    """Generate a mutated set of individuals based on the current
    population. The idea is to sample three individuals (a, b and c)
    from the population and use them to create a mutated individual
    t = a + F * (b - c).

    This implementation follows the definition in:
    https://doi.org/10.1007/11539117_147

    Args:
        population (np.ndarray): float array with shape (n, n_dims).
        F (float | np.ndarray): scaling factor in [0, 2] or array
            with shape (n,).
        seed (int): random seed.
        n (int, optional): number of mutated vectors. 
            Defaults to population size.

    Returns:
        array of mutated solutions with shape (n_mutated, n_dims).
    """
    rng = np.random.default_rng(seed)

    if not isinstance(F, np.ndarray):
        F = np.array(F, dtype=np.float32)

    if n is None:
        n = population.shape[0]

    # Generating all parameters for the
    #   perturbation.

    # The idea is to generate a random matrix
    #   with size (n, pop_size) and obtain the indices
    #   that would yield the sorted array.
    random_matrix = rng.random(size=(n, population.shape[0]))
    random_indices = random_matrix.argsort(axis=1)

    # From those indices, we can select only the first 3
    #   and obtain the actual individuals to be used
    #   as parameters for the mutation.
    parameters_indices = random_indices[:, :3]
    parameters = population[parameters_indices]

    # Extracting the values of each parameter
    #   in 3 arrays of shape (n, d)
    a = parameters[:, 0]
    b = parameters[:, 1]
    c = parameters[:, 2]

    # Calculating the mutated individuals
    #   based on the parameters and F
    F = np.expand_dims(F, axis=-1)
    mutated = a + F * (b - c)

    # Clip new individuals to domain, if
    #   available.
    if bounds is not None:
        mutated = np.clip(mutated, *bounds)

    return mutated


def rand_individual_best_perturbation(population: np.ndarray,
                                      population_fitness: np.ndarray,
                                      F: float | np.ndarray,
                                      seed: int,
                                      n: int = None,
                                      bounds: tuple = None) -> np.ndarray:
    """Generate a mutated set of individuals based on the current
    population and the best known individual.

    This implementation follows the Eq. (3) in:
    https://doi.org/10.1109/CEC.2008.4630935

    Args:
        population (np.ndarray): float array with shape (n, n_dims).
        population_fitness (np.ndarray): float array with shape (n,).
        F (float | np.ndarray): scaling factor in [0, 2] or array
            with shape (n,).
        seed (int): random seed.
        n (int, optional): number of mutated vectors. 
            Defaults to population size.

    Returns:
        array of mutated solutions with shape (n_mutated, n_dims).
    """
    rng = np.random.default_rng(seed)

    if not isinstance(F, np.ndarray):
        F = np.array(F, dtype=np.float32)

    if n is None:
        n = population.shape[0]

    # Generating all parameters for the
    #   perturbation.

    # The idea is to generate a random matrix
    #   with size (n, pop_size) and obtain the indices
    #   that would yield the sorted array.
    random_matrix = rng.random(size=(n, population.shape[0]))
    random_indices = random_matrix.argsort(axis=1)

    # As an intermediate step, we guarantee that the
    #   no row contains its own index (i.e., we don't take
    #   the same individual as a parameter)
    all_indices = np.expand_dims(np.arange(population.shape[0]),
                                 axis=-1)
    is_valid = random_indices[:,] != all_indices
    random_indices = random_indices[is_valid].reshape(
        (n, population.shape[0] - 1))

    # From those indices, we can select only the first 2
    #   and obtain the actual individuals to be used
    #   as parameters for the mutation.
    parameters_indices = random_indices[:, :2]
    parameters = population[parameters_indices]

    # Extracting the values of each parameter
    #   in 2 arrays of shape (n, d)
    b = parameters[:, 0]
    c = parameters[:, 1]

    # Additionally, we need the best individual
    #   of the population
    best_idx = population_fitness.argmin()
    best_ind = population[best_idx]

    # Calculating the mutated individuals
    #   based on the parameters, best individual and F:
    #   ci = xi + F(best_ind - xi) + F(b - c)
    F = np.expand_dims(F, axis=-1)
    mutated = population + \
        F * (best_ind + - population) + \
        F * (b - c)

    # Clip new individuals to domain, if
    #   available.
    if bounds is not None:
        mutated = np.clip(mutated, *bounds)

    return mutated
