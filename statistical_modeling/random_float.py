from . import (
    ContinuousDistribution,
    ContinuousDistributionAlgorithm,
    DefaultAlgorithm
)
from typing import TypeVar
from final_class import final

T = TypeVar('T', bound=ContinuousDistribution)

__all__ = [
    'RandomFloat',
]


@final
class RandomFloat(float):
    __distribution: T
    __algorithm: ContinuousDistributionAlgorithm[T]

    def __new__(
        cls, distribution: T, algorithm: ContinuousDistributionAlgorithm[T] = None
    ) -> 'RandomFloat':
        if algorithm is not None:
            return super().__new__(cls, cls.__value(distribution, algorithm))
        else:
            return super().__new__(cls, cls.__value(distribution, DefaultAlgorithm(distribution)))

    def __init__(self, distribution: T, algorithm: ContinuousDistributionAlgorithm[T] = None):
        self.__distribution = distribution
        if algorithm is not None:
            self.__algorithm = algorithm
        else:
            self.__algorithm = DefaultAlgorithm(distribution)

    @staticmethod
    def __value(
        distribution: ContinuousDistribution, algorithm: ContinuousDistributionAlgorithm
    ) -> float:
        return algorithm.value(distribution)

    @property
    def distribution(self):
        return self.__distribution

    @property
    def algorithm(self):
        return self.__algorithm
