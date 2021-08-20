from abc import ABC, abstractmethod
from typing import TypeVar, Generic

__all__ = ['Distribution',
           'DiscreteDistribution',
           'ContinuousDistribution',
           'DistributionAlgorithm',
           'DiscreteDistributionAlgorithm',
           'ContinuousDistributionAlgorithm',
           'DistributionMean',
           'DistributionVariance']


class Distribution(ABC):
    pass


class DiscreteDistribution(Distribution):
    pass


class ContinuousDistribution(Distribution):
    pass


T = TypeVar('T')
U = TypeVar('U')


class DistributionAlgorithm(ABC, Generic[T, U]):
    @abstractmethod
    def value(self, distribution: T) -> U:
        pass


class DiscreteDistributionAlgorithm(DistributionAlgorithm[T, int], Generic[T]):
    @abstractmethod
    def value(self, distribution: T) -> int:
        pass


class ContinuousDistributionAlgorithm(DistributionAlgorithm[T, float], Generic[T]):
    @abstractmethod
    def value(self, distribution: T) -> float:
        pass


class DistributionMean(ABC, float):
    __distribution: Distribution

    def __init__(self, distribution: Distribution):
        self.__distribution = distribution

    @property
    def distribution(self) -> Distribution:
        return self.__distribution


class DistributionVariance(ABC, float):
    __distribution: Distribution

    def __init__(self, distribution: Distribution):
        self.__distribution = distribution

    @property
    def distribution(self) -> Distribution:
        return self.__distribution
