"""Decomposition strategies.
"""
from __future__ import annotations

import numpy as np


def random_group_decompose(dims: int,
                           n_groups: int,
                           seed: int) -> list[np.ndarray]:
    """Decompose the decisions variables into
    random groups with approximately `group_dim` 
    dimensions.

    Args:
        dims (int): total number of dimensions.
        n_groups (int): number of groups to generate.
            Each generated group has roughly 
            dims / n_groups dimensions.
        seed (int): random seed.

    Returns:
        list of arrays.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dims)
    splits = np.array_split(permutation, n_groups)
    return splits

def dimension_decompose(dims: int) -> np.ndarray:
    """Decompose the decisions variables into
    groups of size 1 for each dimension.

    Args:
        dims (int): number of dimensions.

    Returns:
        indices for each group with shape
            (n_dims, 1).
    """
    return np.expand_dims(np.arange(dims), axis=-1)


def half_decompose(dims: int) -> tuple[np.ndarray,
                                       np.ndarray]:
    """Decompose the decisions variables into to
    groups of roughly equal size.

    Args:
        dims (int): number of dimensions.

    Returns:
        tuple of arrays (first_half, second_half).
    """
    indices = np.arange(dims)
    mid_point = dims // 2
    return indices[:mid_point], indices[mid_point:]
