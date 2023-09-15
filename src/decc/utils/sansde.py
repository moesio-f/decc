"""Utility for evolving a
population with the SaNSDE
algorithm.

Reference
---------
Zhenyu Yang, Ke Tang and Xin Yao.
Self-adaptive differential evolution 
    with neighborhood search (2008) 
https://doi.org/10.1109/CEC.2008.4630935
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from decc.operators import crossover, mutation
from decc.operators import neighborhood_search as ns
from decc.operators import selection


def sansde(population: np.ndarray,
           fitness: np.ndarray,
           fn: Callable[[np.ndarray], np.ndarray],
           seed: int,
           mp: float = 0.5,
           fp: float = 0.5,
           crm: float = 0.5,
           n_gens_cr_update: int = 5,
           n_gens_prob_update: int = 50,
           n_gens_crm_update: int = 25,
           max_fn: int | None = None,
           bounds: tuple | None = None) -> tuple[np.ndarray,
                                                 np.ndarray,
                                                 int]:
    if max_fn is None:
        max_fn = population.shape[0]

    # Auxiliary variables
    n_evaluations = 0
    CR_h: list[np.ndarray] = []
    CR_ih = []
    nsm1, nsm2 = 0, 0  # No. Success of Mutation method i
    nfm1, nfm2 = 0, 0  # No. Fail of Mutation method i
    nsf1, nsf2 = 0, 0  # No. Success of F method i
    nff1, nff2 = 0, 0  # No. Fail of F method i
    learning_gens_prob = 1
    learning_gens_crm = 1
    last_cr_update = 0
    generation = 0
    pop_size = population.shape[0]
    dims = population.shape[-1]
    seed_interval = (0, 99999)

    # Generator
    rng = np.random.default_rng(seed)

    # Initial CR
    CR = rng.normal(crm, 0.1, size=(pop_size,))

    # Evolution loop
    while n_evaluations < max_fn:
        # Applying NS to obtain values of F
        #   for each individual in the population
        F_seed = rng.integers(*seed_interval)
        F, F_where_gaussian = ns.random_gaussian_cauchy(
            fp=fp,
            seed=F_seed,
            n=pop_size,
            g_loc=0.5,
            g_scale=0.3)

        # Mutation information
        mutation_seed_m1 = rng.integers(*seed_interval)
        mutation_seed_m2 = rng.integers(*seed_interval)

        # Find mutated population with method 1
        mutated_m1 = mutation.rand_individual_pertubation(
            population,
            F,
            mutation_seed_m1,
            bounds=bounds)

        # Find mutated population with method 2
        mutated_m2 = mutation.rand_individual_best_perturbation(
            population,
            fitness,
            F,
            mutation_seed_m2,
            bounds=bounds)

        # Obtaining mutation probabilities
        mutation_prob = rng.uniform(0, 1, size=(pop_size,))

        # Obtain the mutated population using
        #   the probability of select individuals
        #   from each method.
        mutation_cond = np.expand_dims(mutation_prob < mp,
                                       axis=-1)
        mutation_cond = np.broadcast_to(mutation_cond,
                                        shape=(pop_size, dims))
        mutated_pop = np.where(mutation_cond,
                               mutated_m1,
                               mutated_m2)

        # Crossover information
        if (generation - last_cr_update) >= n_gens_cr_update:
            CR = rng.normal(crm, 0.1, size=(pop_size,))

        # Applying Crossover
        candidates = crossover.dim_prob_crossover(
            population,
            mutated_pop,
            CR,
            rng.integers(*seed_interval))

        # Candidate evaluation
        candidates_fitness = fn(candidates)
        n_evaluations += candidates.shape[0]

        # Update population based on fitness
        old_population, old_fitness = population, fitness
        population, fitness = selection.best_fitness_selection(
            population,
            candidates,
            fitness,
            candidates_fitness)

        # === Update CR history ===
        where_new = np.any(population != old_population,
                           axis=1)
        fitness_diff = np.abs(fitness - old_fitness)
        CR_h.append(np.copy(CR[where_new]))
        CR_ih.append(np.copy(fitness_diff[where_new]))
        assert len(CR_h[-1]) == len(CR_ih[-1])
        del where_new, fitness_diff

        # === Update running metrics for mp and fp ===
        # --- Mutation Method 1 ---
        successes = np.all(population == mutated_m1,
                           axis=1)
        n = np.count_nonzero(successes)
        nsm1 += n
        nfm1 += len(mutated_m1) - n
        f_metrics = get_F_metrics(successes,
                                  F_where_gaussian)
        nsf1 += f_metrics[0]
        nsf2 += f_metrics[1]
        nff1 += f_metrics[2]
        nff2 += f_metrics[3]
        del successes, n, f_metrics

        # --- Mutation Method 2 ---
        successes = np.all(population == mutated_m2,
                           axis=1)
        n = np.count_nonzero(successes)
        nsm2 += n
        nfm2 += len(mutated_m2) - n
        f_metrics = get_F_metrics(successes,
                                  F_where_gaussian)
        nsf1 += f_metrics[0]
        nsf2 += f_metrics[1]
        nff1 += f_metrics[2]
        nff2 += f_metrics[3]
        del successes, n, f_metrics

        # === Self-updating parameters mp, fp, and crm ===
        # Mutation probability (mp) should be updated
        #   once N generations have passed
        if learning_gens_prob >= n_gens_prob_update:
            term1 = nsm1 * (nsm2 + nfm2)
            term2 = nsm2 * (nsm1 + nfm1)
            try:
                mp = term1 / (term1 + term2)
            except ZeroDivisionError:
                mp = 0.5

            # Reset history
            nsm1, nsm2 = 0, 0
            nfm1, nfm2 = 0, 0

        # F probability (fp) should be updated
        #   once N generations have passed
        if learning_gens_prob >= n_gens_prob_update:
            term1 = nsf1 * (nsf2 + nff2)
            term2 = nsf2 * (nsf1 + nff1)
            try:
                fp = term1 / (term1 + term2)
            except ZeroDivisionError:
                fp = 0.5

            # Reset history
            nsf1, nsf2 = 0, 0
            nff1, nff2 = 0, 0
            learning_gens_prob = 1

        # Crossover mean (CRm) should be updated
        #   once N generations have passed
        if learning_gens_crm >= n_gens_crm_update:
            # Converting records to np arrays
            CR_r = np.concatenate(CR_h)
            CR_i = np.concatenate(CR_ih)
            CR_w = CR_i / CR_i.sum()

            # Updating based on the relative
            #   improvement (weights)
            crm = np.sum(CR_r * CR_w)

            # Reset the history
            CR_h = []
            CR_ih = []
            learning_gens_crm = 1

        # === Generation finished ===
        learning_gens_prob += 1
        learning_gens_crm += 1
        generation += 1

    return population, fitness, n_evaluations


def get_F_metrics(successes: np.ndarray,
                  where_gaussian: np.ndarray) -> tuple[int, int,
                                                       int, int]:
    """Obtain the metrics of successes and fails
    for each F strategy (Gaussian or Cauchy).

    Args:
        successes (np.ndarray): boolean array where each
            entry represents whether the individual was
            a success (i.e., was selected to new generation).
        where_gaussian (np.ndarray): boolean array that indicates
            if the Gaussian strategy was used.

    Returns:
        (ns1, ns2, nf1, nf2)
    """
    # From the success individuals of this
    #   mutation, we can find the successes
    #   of the F strategy (Gaussian or Cauchy).
    n1 = np.count_nonzero(np.logical_and(successes,
                                         where_gaussian))
    n2 = np.count_nonzero(np.logical_and(successes,
                                         ~where_gaussian))

    # Likewise, we can obtain the fails
    n3 = np.count_nonzero(np.logical_and(~successes,
                                         where_gaussian))
    n4 = np.count_nonzero(np.logical_and(~successes,
                                         ~where_gaussian))

    return n1, n2, n3, n4
