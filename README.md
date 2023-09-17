# decc: Cooperative Co-evolutionary Differential Evolution

Cooperative Co-evolutionary Differential Evolution algorithms in Python with NumPy. 

This repository provides simple implementations for some CC-based DE algorithms. All the implementations are from scratch using NumPy, they might slightly differ from the original papers.

# Quick Start

The `decc` library works with two entities:

- `Problem` represents a minimization problem (objective function, dimensionality and bounds).
  - The objective function must receives a matrix as input (each row represents a solution) and produce a matrix as output (each row is the objective value for the respective solution).
- `Optimizer` represents a DE algorithm for solving a minimization problem.
  - Maximization of a real function `g` is equivalent to minimize `-g`.

In order to use an optimizer, you must first define a problem. Currently, there are two optimizer available: (i) [**DECC**](src/decc/optimizers/decc.py) (also known as CCDE), with the variants **DECC-O** and **DECC-H**; and (ii) [**DECC-G**](src/decc/optimizers/decc_g.py). A basic example is found below:

```python
import numpy as np

from decc import Problem
from decc.optimizers.decc import DECCOptimizer
from decc.optimizers.decc_g import DECCGOptimizer

def objective(x: np.ndarray) -> np.ndarray:
    # The Sphere function is a common
    #   benchmark for optimizers
    return np.sum(x ** 2, axis=-1)


# First, we must define the problem
problem = Problem(objective,
                  dims=100,
                  lower_bound=-100.0,
                  upper_bound=100.0)

# Then, we can instantiate the optimizers
F = 0.5
CR = 0.8
seed = 42
max_fn = int(1e5)
pop_size = 50
decc_h = DECCOptimizer(problem,
                       seed,
                       pop_size,
                       grouping='halve',
                       F=F,
                       CR=CR,
                       max_fn=max_fn)
decc_o = DECCOptimizer(problem,
                       seed,
                       pop_size,
                       grouping='dims',
                       F=F,
                       CR=CR,
                       max_fn=max_fn)
decc_g = DECCGOptimizer(problem,
                        seed,
                        pop_size,
                        n_subproblems=max(1, problem.dims // 4),
                        sansde_evaluations=max_fn // 3,
                        de_evaluations=max_fn // 5,
                        F=F,
                        CR=CR,
                        max_fn=max_fn)

# Lastly, we can optimize the objective and
#   retrieve the results.
for optimizer in [decc_o, decc_h, decc_g]:
    result = optimizer.optimize()
    print(f'{optimizer}: {result["best_fitness"]}')

# DECC-O: 1.638248431845568e-05
# DECC-H: 0.0006988301174715161
# DECC-G: 1.5340782547163752e-10
```

# Roadmap

- Add support for `DECC-DG`;
- Add support for `DECC-gDG`;
- Improve documentation;
- Add unit tests;
- Add validation benchmark functions from the original papers;
