"""DECC optimizer.

Reference
---------
Shi, Yj., Teng, Hf., Li, Zq.
Cooperative Co-evolutionary Differential Evolution 
    for Function Optimization (2005) 
https://doi.org/10.1007/11539117_147
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from decc import decomposition
from decc.core import Optimizer, Problem
from decc.utils import classic_de as de


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
    
    def name(self) -> str:
        return self.variant

    def _optimize(self, *args, **kwargs) -> tuple[np.ndarray,
                                                  np.ndarray,
                                                  dict | None]:
        del args, kwargs

        # Variables
        rng = np.random.default_rng(self.seed)
        l, u = self.problem.bounds
        d = self.problem.dims
        p = self.pop_size
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
        best_solution = context_vector
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

            # Update the population using the context (i. e.,
            #   set the indices that won't be evolved to the
            #   same value as the context)
            # New shape (n_subpopulation, n_dims)
            population = self._population_w_context(
                context_vector,
                population,
                indices)

            # Obtaining the population fitness
            population_fitness = fn(population)
            n_evaluations += sp

            # Calculating best
            update_best(population, population_fitness)

            # Storing the subpopulation and its fitness
            subpopulations.append(population)
            subpopulations_fitness.append(population_fitness)

        # Evolution loop
        while n_evaluations <= self.max_fn:
            # Obtaining the new context vector
            context_vector = np.zeros((d,),
                                      dtype=np.float32)

            for i, indices in enumerate(self.subproblem_indices):
                best_idx = subpopulations_fitness[i].argmin()
                context_vector[indices] = subpopulations[i][best_idx, indices]

            # Evaluating the new fitness
            context_fitness = fn(np.expand_dims(context_vector,
                                                axis=0))

            # Updating best
            if context_fitness < best_fitness:
                best_solution = context_vector
                best_fitness = context_fitness

            # Evolve each subpopulation
            for i, indices in enumerate(self.subproblem_indices):
                # Obtaining current information
                population = subpopulations[i]
                fitness = subpopulations_fitness[i]
                l_, u_ = l[indices], u[indices]

                # Obtaining the population to evolve by selecting
                #   only the indices (dimensions) to be optimized
                #   by DE.
                evolvable_population = population[:, indices]

                # Creating a fitness function which
                #   maps the evolvable population (contains
                #   solutions with smaller dimension)
                #   back to the actual population.
                def _fn(pop: np.ndarray) -> np.ndarray:
                    full_dims_pop = np.copy(population)
                    full_dims_pop[:, indices] = pop
                    return fn(full_dims_pop)

                # Apply DE to obtain new population
                pop, fitness, n = de.de_rand_1_exp(
                    population=evolvable_population,
                    fitness=fitness,
                    F=self.F,
                    CR=self.CR,
                    fn=_fn,
                    seed=rng.integers(0, 9999),
                    bounds=(l_, u_))
                n_evaluations += n

                # Update the actual population
                population[:, indices] = pop
                subpopulations[i] = population
                subpopulations_fitness[i] = fitness

                # Updating best
                update_best(subpopulations[i],
                            subpopulations_fitness[i])

        return best_fitness[0], best_solution, None

    def _population_w_context(self,
                              context: np.ndarray,
                              population: np.ndarray,
                              indices: np.ndarray) -> np.ndarray:
        # Create context matrix from
        #   context vector.
        context_population = np.copy(np.broadcast_to(
            context,
            shape=(population.shape[0],
                   self.problem.dims)))

        # Update values in indices
        context_population[:, indices] = population

        return context_population
