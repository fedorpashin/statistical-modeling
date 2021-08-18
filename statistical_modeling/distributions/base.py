from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from multimethod import multimeta

__all__ = ['Distribution',
           'DiscreteDistribution',
           'ContinuousDistribution',
           'DistributionAlgorithm',
           'DiscreteDistributionAlgorithm',
           'ContinuousDistributionAlgorithm']


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


# @todo #6:120min Implement factory classes for quantities

# @todo #6:120min Make appropriate classes derived from a number class (int or float)

# @todo #6:30min Modify README

# @todo #22:15min Make Algorithm classes abstract
