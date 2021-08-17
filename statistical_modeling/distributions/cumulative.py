from typing import Callable

from numpy import random


def cumulative(p_of: Callable[[int], float], x: int = 0) -> int:
    m = random.uniform()
    while True:
        m -= p_of(x)
        if m < 0:
            break
        x += 1
    return x
