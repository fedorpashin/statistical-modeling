from . import (
    DiscreteDistribution,
    DiscreteDistributionAlgorithm,
    DefaultAlgorithm
)
from typing import TypeVar
from final_class import final

T = TypeVar('T', bound=DiscreteDistribution)

__all__ = [
    'RandomInt',
]


@final
class RandomInt(int):
    __distribution: T
    __algorithm: DiscreteDistributionAlgorithm[T]

    def __new__(
        cls, distribution: T, algorithm: DiscreteDistributionAlgorithm[T] = None
    ) -> 'RandomInt':
        if algorithm is not None:
            return super().__new__(cls, cls.__value(distribution, algorithm))
        else:
            return super().__new__(cls, cls.__value(distribution, DefaultAlgorithm(distribution)))

    def __init__(self, distribution: T, algorithm: DiscreteDistributionAlgorithm[T] = None):
        self.__distribution = distribution
        if algorithm is not None:
            self.__algorithm = algorithm
        else:
            self.__algorithm = DefaultAlgorithm(distribution)

    @staticmethod
    def __value(
        distribution: DiscreteDistribution, algorithm: DiscreteDistributionAlgorithm
    ) -> int:
        return algorithm.value(distribution)

    @property
    def distribution(self):
        return self.__distribution

    @property
    def algorithm(self):
        return self.__algorithm
