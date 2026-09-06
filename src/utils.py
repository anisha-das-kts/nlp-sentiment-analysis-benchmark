from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timer():
    """
    Context manager for measuring elapsed time.

    Example
    -------
    with timer() as elapsed:
        # work
        pass

    print(elapsed())
    """
    start = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - start

    yield elapsed