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


class DistributionAlgorithm(ABC, Generic[T]):
    @staticmethod
    @abstractmethod
    def default(distribution: T) -> 'DistributionAlgorithm[T]':
        pass


class DiscreteDistributionAlgorithm(DistributionAlgorithm[T], Generic[T]):
    @abstractmethod
    def value(self, distribution: T) -> int:
        pass


class ContinuousDistributionAlgorithm(DistributionAlgorithm[T], Generic[T]):
    @abstractmethod
    def value(self, distribution: T) -> float:
        pass


class DistributionMean(ABC, float):
    __distribution: Distribution

    def __init__(self, distribution: Distribution):
        self.__distribution = distribution

    @staticmethod
    @abstractmethod
    def __value(distribution: Distribution) -> float:
        pass

    @property
    def distribution(self) -> Distribution:
        return self.__distribution


class DistributionVariance(ABC, float):
    __distribution: Distribution

    def __init__(self, distribution: Distribution):
        self.__distribution = distribution

    @staticmethod
    @abstractmethod
    def __value(distribution: Distribution) -> float:
        pass

    @property
    def distribution(self) -> Distribution:
        return self.__distribution


# @todo #6:120min Implement factory classes for quantities

# @todo #6:30min Modify README

# @todo #22:15min Make Algorithm classes abstract
