from abc import ABC, abstractmethod
from typing import TypeVar, Generic

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
    @classmethod
    @abstractmethod
    def default(cls, distribution: T) -> 'DistributionAlgorithm[T]':
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
