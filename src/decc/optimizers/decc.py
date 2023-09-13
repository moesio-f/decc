"""DECC optimizer.

Reference
---------
Shi, Yj., Teng, Hf., Li, Zq. (2005). 
Cooperative Co-evolutionary Differential Evolution 
    for Function Optimization (2005) 
https://doi.org/10.1007/11539117_147
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from decc import decomposition
from decc.core import Optimizer
from decc.core.problem import Problem
from decc.operators import crossover, mutation


class DECCOptimizer(Optimizer):
    def __init__(self,
                 problem: Problem,
                 seed: int,
                 subpopulation_size: int,
                 max_fn: int = int(1e6),
                 grouping: Literal['halve',
                                   'dims'] = 'halve',
                 F: float = 0.8,
                 CR: float = 0.3) -> None:
        super().__init__(problem, seed)
        self.subpop_size = subpopulation_size
        self.F = F
        self.CR = CR
        self.max_fn = max_fn
        self.variant = 'DECC-'

        d = self.problem.dims
        if grouping == 'halve':
            self.variant += 'H'
            self.subproblem_indices = decomposition.half_decompose(d)
        else:
            self.variant += 'O'
            self.subproblem_indices = decomposition.dimension_decompose(d)

        self.pop_size = self.subpop_size * len(self.subproblem_indices)

    def parameters(self) -> dict:
        return {
            'variant': self.variant,
            'population_size': self.pop_size,
            'subpopulation_size': self.subpop_size,
            'F': self.F,
            'CR': self.CR,
            'max_evaluations': self.max_fn,
            'n_subproblems': len(self.subproblem_indices)
        }

    def _optimize(self, *args, **kwargs) -> tuple[np.ndarray,
                                                  np.ndarray,
                                                  dict | None]:
        del args, kwargs

        # Variables
        rng = np.random.default_rng(self.seed)
        l, u = self.problem.bounds
        d = self.problem.dims
        sp = self.subpop_size
        fn = self.problem.fn
        n_evaluations = 0
        best_solution = None
        best_fitness = None

        def update_best(population: np.ndarray,
                        fitness: np.ndarray):
            nonlocal best_fitness
            nonlocal best_solution

            best_idx = fitness.argmin()
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]

        # Initializing context vector
        context_vector = rng.uniform(l, u, size=(1, d))
        context_fitness = fn(context_vector)

        # Updating variables
        best_solution = context_vector[0]
        best_fitness = context_fitness
        n_evaluations += 1

        # Initializing subpopulations
        subpopulations = []
        subpopulations_fitness = []

        for indices in self.subproblem_indices:
            # Generating the population with shape
            #   (n_subpopulation, n_dims_subproblem)
            l_, u_ = l[indices], u[indices]
            population = rng.uniform(l_, u_,
                                     size=(sp, len(indices)))

            # Obtaining the complete representation
            #   of the population using the context vector
            context_population = self._individuals_from_decomposed(context_vector,
                                                                   population,
                                                                   indices)

            # Obtaining the population fitness
            population_fitness = fn(context_population)
            n_evaluations += sp

            # Calculating best
            update_best(context_population, population_fitness)

            # Storing the subpopulation and its fitness
            subpopulations.append(population)
            subpopulations_fitness.append(population_fitness)

        # Evolution loop
        while n_evaluations <= self.max_fn:
            # Obtaining the new context vector
            context_vector = np.concatenate([sp[spf.argmin()]
                                             for sp, spf in zip(subpopulations,
                                                                subpopulations_fitness)],
                                            dtype=np.float32)
            context_fitness = fn(np.expand_dims(context_vector, axis=0))

            # Updating best
            if context_fitness < best_fitness:
                best_solution = context_vector
                best_fitness = context_fitness

            # Evolve each subpopulation
            for i, indices in enumerate(self.subproblem_indices):
                # Obtaining current population information
                current_fitness = subpopulations_fitness[i]
                current_population = subpopulations[i]

                # Mutation
                mutated_pop = mutation.rand_individual_pertubation(
                    current_population,
                    self.F,
                    rng.integers(0, 99999))

                # Crossover
                candidates = crossover.dim_prob_crossover(
                    current_population,
                    mutated_pop,
                    self.CR,
                    rng.integers(0, 99999))

                # Obtaining individuals using new
                #   representative
                context_candidates = self._individuals_from_decomposed(
                    context_vector,
                    candidates,
                    indices
                )

                # Candidate evaluation
                candidates_fitness = fn(context_candidates)
                n_evaluations += sp

                # Update population based on fitness
                cond = candidates_fitness < current_fitness
                cond_dims = np.repeat(np.expand_dims(cond, axis=-1),
                                      candidates.shape[-1],
                                      axis=-1)
                subpopulations[i] = np.where(cond_dims,
                                             candidates,
                                             current_population)
                subpopulations_fitness[i] = np.where(cond,
                                                     candidates_fitness,
                                                     current_fitness)

        return best_fitness, best_solution, None

    def _individuals_from_decomposed(self,
                                     context: np.ndarray,
                                     individuals: np.ndarray,
                                     indices: np.ndarray) -> np.ndarray:
        # Create context matrix from
        #   context vector.
        context_individuals = np.copy(np.broadcast_to(
            context,
            shape=(individuals.shape[0],
                   self.problem.dims)))

        # Update values in indices
        context_individuals[:, indices] = individuals

        return context_individuals
