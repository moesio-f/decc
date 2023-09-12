"""Problem interface definition.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class Problem:
    def __init__(self) -> None:
        pass

    @property
    def fn(self) -> Callable[[np.ndarray], np.ndarray]:
        pass

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    def dims(self) -> int: ...
