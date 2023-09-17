"""DECC-G optimizer.

Reference
---------
Yang, Z., Tang, K., & Yao, X.
Large scale evolutionary optimization 
    using cooperative coevolution (2008) 
https://doi.org/10.1016/j.ins.2008.02.017 
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from decc import decomposition
from decc.core import Optimizer, Problem
from decc.utils import classic_de as de
from decc.utils import sansde


class DECCGOptimizer(Optimizer):
    def __init__(self,
                 problem: Problem,
                 seed: int,
                 population_size: int,
                 n_subproblems: int,
                 sansde_evaluations: int,
                 de_evaluations: int,
                 max_fn: int = int(1e6),
                 F: float = 0.8,
                 CR: float = 0.3,
                 weights_bound: tuple[float, float] = (-5, 5)) -> None:
        super().__init__(problem, seed)
        assert n_subproblems <= self.problem.dims

        self.pop_size = population_size
        self.F = F
        self.CR = CR
        self.max_fn = max_fn
        self.n_sub = n_subproblems
        self.de_eval = de_evaluations
        self.sansde_eval = sansde_evaluations
        self.w_bounds = weights_bound

    def parameters(self) -> dict:
        return {
            'population_size': self.pop_size,
            'DE': {
                'F': self.F,
                'CR': self.CR,
                'evaluations': self.de_eval
            },
            'SaNSDE': {
                'evaluations': self.sansde_eval
            },
            'max_evaluations': self.max_fn,
            'n_subproblems': self.n_sub
        }
    
    def name(self) -> str:
        return "DECC-G"

    def _optimize(self, *args, **kwargs) -> tuple[np.ndarray,
                                                  np.ndarray,
                                                  dict | None]:
        del args, kwargs

        # Variables
        rng = np.random.default_rng(self.seed)
        l, u = self.problem.bounds
        d = self.problem.dims
        p = self.pop_size
        fn = self.problem.fn
        n_evaluations = 0
        best_solution = None
        best_fitness = None
        seed_interval = (0, 99999)

        # Random initial population of
        #   shape (pop_size, dims)
        population = rng.uniform(l, u, size=(p, d))
        population_fitness = fn(population)
        n_evaluations += p

        # Initialize weight population of
        #   shape (pop_size, n_subproblems)
        #   withe zeroes
        weight_population = np.zeros(shape=(p, self.n_sub),
                                     dtype=np.float32)

        # Function to update best
        def update_best(population: np.ndarray,
                        fitness: np.ndarray):
            nonlocal best_fitness
            nonlocal best_solution

            best_idx = fitness.argmin()
            empty = best_fitness is None
            if empty or fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]

        # Update best
        update_best(population, population_fitness)

        while n_evaluations <= self.max_fn:
            # Generating random groups
            group_seed = rng.integers(*seed_interval)
            groups = decomposition.random_group_decompose(
                d,
                self.n_sub,
                seed=group_seed)

            # Evolving subpopulations
            for i, g in enumerate(groups):
                # Obtaining sub-population
                sub_pop = population[:, g]
                l_, u_ = l[g], u[g]

                # Creating a fitness function which
                #   maps the sub-population (contains
                #   solutions with smaller dimension)
                #   back to the actual population.
                def _fn(pop: np.ndarray) -> np.ndarray:
                    full_dims_pop = np.copy(population)
                    full_dims_pop[:, g] = pop
                    return fn(full_dims_pop)

                # Apply SaNSDE to obtain new population
                sub_pop, fitness, n = sansde.sansde(
                    population=sub_pop,
                    fitness=population_fitness,
                    fn=_fn,
                    seed=rng.integers(*seed_interval),
                    max_fn=self.sansde_eval,
                    bounds=(l_, u_))
                n_evaluations += n

                # Generating random weights for this
                #   group.
                weight_population[:, i] = rng.uniform(*self.w_bounds,
                                                      size=(p,))

                # Updating population and fitness
                population[:, g] = sub_pop
                population_fitness[:] = fitness

                # Update best
                update_best(population, population_fitness)

        # === Updating weight population ===
        def _fn(w_pop: np.ndarray,
                idx: int):
            # Obtaining the individual
            ind = population[idx]
            ind = np.broadcast_to(
                ind,
                shape=(w_pop.shape[0],
                       ind.shape[-1]))
            ind = np.copy(ind)

            # Applying the weights
            for i, g in enumerate(groups):
                # ind[:, g] has shape (pop_size, dims_subproblem)
                sub_ind = ind[:, g]

                # w_pop[:, i] has shape (pop_size,)
                # Broadcasting the weights to
                #   (pop_size, dims_subproblem)
                weights = np.expand_dims(w_pop[:, i], axis=-1)
                weights = np.broadcast_to(weights, sub_ind.shape)

                # Updating with new weighted individuals
                ind[:, g] = weights * sub_ind

            # Evaluating the weighted population
            return fn(ind)

        def _update_pop_w_weights(idx: int):
            nonlocal n_evaluations
            nonlocal population

            # Obtain initial fitness of
            #   weighted individuals
            w_f = _fn(weight_population, idx)
            n_evaluations += p

            # Apply DE to obtain new weight population
            w_pop, w_f, n = de.de_rand_1_exp(
                population=weight_population,
                fitness=w_f,
                F=self.F,
                CR=self.CR,
                fn=lambda pop: _fn(pop, idx),
                seed=rng.integers(*seed_interval),
                bounds=self.w_bounds)
            n_evaluations += n

            # Obtain new weighted individual
            best_w_idx = w_f.argmin()
            best_w = w_pop[best_w_idx]
            best_w_f = w_f[best_w_idx]
            weighted_individual = np.zeros((d,),
                                           dtype=np.float32)

            for i, g in enumerate(groups):
                weighted_individual[g] = best_w[i] * population[idx, g]

            # Update population in this idx
            population[idx] = weighted_individual
            population_fitness[idx] = best_w_f

        # -- Best --
        _update_pop_w_weights(population_fitness.argmin())

        # -- Worst --
        _update_pop_w_weights(population_fitness.argmax())

        # -- Random --
        _update_pop_w_weights(rng.integers(0, len(population)))

        # Update best
        update_best(population, population_fitness)

        return best_fitness, best_solution, None
