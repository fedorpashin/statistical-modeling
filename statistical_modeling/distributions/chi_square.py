from .base import ContinuousDistribution, ContinuousDistributionAlgorithm
from abc import abstractmethod

from numpy import random

from functools import cached_property


class Distribution(ContinuousDistribution):
    __n: int

    def __init__(self, n: int):
        assert n > 0
        self.__n = n

    @property
    def n(self) -> int:
        return self.__n


class Algorithm(ContinuousDistributionAlgorithm['Algorithm']):
    @classmethod
    def default(cls, distribution: Distribution) -> 'Algorithm':
        return StandardAlgorithm()

    @abstractmethod
    def value(self, distribution: Distribution) -> float:
        pass


class StandardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        n = distribution.n
        return sum([random.standard_normal()**2 for _ in range(n)])


class Mean:
    __distribution: Distribution

    @cached_property
    def __value(self) -> float:
        n = self.__distribution.n
        return n


class Variance:
    __distribution: Distribution

    @cached_property
    def __value(self) -> float:
        n = self.__distribution.n
        return 2 * n
