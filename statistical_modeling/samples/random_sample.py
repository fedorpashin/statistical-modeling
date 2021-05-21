from .sample import *
from ..distributions.base import *

from typing import Optional

__all__ = ['RandomSample']


class RandomSample(Sample):
    _x: list[float]

    _distribution: Distribution
    _algorithm: Optional[DistributionAlgorithm]

    @property
    def distribution(self):
        return self._distribution

    @property
    def algorithm(self):
        return self._algorithm

    def __init__(self, distribution: Distribution, n: int, algorithm: Optional[DistributionAlgorithm] = None):
        if algorithm is not None:
            assert isinstance(algorithm, distribution.Algorithm)
        self._distribution = distribution
        self._algorithm = algorithm
        super().__init__([self._distribution.generate(self._algorithm) for _ in range(n)])
