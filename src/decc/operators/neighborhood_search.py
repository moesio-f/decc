"""Neighborhood search operators.
"""
from __future__ import annotations

import numpy as np


def random_gaussian_cauchy(fp: float,
                           seed: int,
                           n: int = 1,
                           g_scale: float = 0.5,
                           g_loc: float = 0.5) -> tuple[np.ndarray,
                                                        np.ndarray]:
    """Generates a vector of random number drawn
    from either a Gaussian or Cauchy distribution.

    This implementation follows the definition in:
    https://doi.org/10.1109/CEC.2008.4630935

    Args:
        fp (float): threshold to choose Cauchy.
        seed (int): random seed.
        n (int, optional): size of the resulting
            array. Defaults to 1.
        g_scale (float, optional): Gaussian distribution
            parameter. Defaults to 0.5.
        g_loc (float, optional): Gaussian distribution
            parameter. Defaults to 0.5.

    Returns:
        (F, cond) arrays with shape (n,). F contains
            the random values, while cond is a boolean
            array indicating where prob < fp.
    """
    rng = np.random.default_rng(seed)

    # Generate random Gaussian values
    gaussian = rng.normal(size=(n,),
                          loc=g_loc,
                          scale=g_scale)

    # Generate random Cauchy values
    cauchy = rng.standard_cauchy(size=(n,))

    # Generate probabilities
    probabilities = rng.uniform(0, 1, size=(n,))

    # Choose whether to select the Gaussian
    #   or Cauchy numbers
    cond = probabilities < fp
    return np.where(cond, gaussian, cauchy), cond
