from .sample import AnySample
from .distributions.base import (
    Distribution,
    DistributionAlgorithm
)
from .factories.default_algorithm import DefaultAlgorithm
from typing import TypeVar
from final_class import final
from overrides import overrides

from functools import cached_property

__all__ = [
    'RandomSample',
]

T = TypeVar('T', bound=Distribution)
U = TypeVar('U')


@final
class RandomSample(AnySample):
    __n: int
    __distribution: T
    __algorithm: DistributionAlgorithm[T, U]

    def __init__(self, n: int, distribution: T, algorithm: DistributionAlgorithm[T, U] = None):
        self.__n = n
        self.__distribution = distribution
        if algorithm is not None:
            self.__algorithm = algorithm
        else:
            self.__algorithm = DefaultAlgorithm(distribution)

    @cached_property  # noqa
    @overrides
    def x(self) -> list[float]:
        return [self.algorithm.value(self.distribution) for _ in range(self.__n)]

    @property
    def distribution(self):
        return self.__distribution

    @property
    def algorithm(self):
        return self.__algorithm
