from .base import ContinuousDistribution, ContinuousDistributionAlgorithm
from abc import abstractmethod

from math import sqrt, inf
from numpy import random

from functools import cached_property


class Distribution(ContinuousDistribution):
    __n: float

    def __init__(self, n: int):
        assert n > 0
        self.__n = n

    @property
    def n(self) -> float:
        return self.__n


class Algorithm(ContinuousDistributionAlgorithm['Algorithm']):
    @staticmethod
    def default(distribution: Distribution) -> 'Algorithm':
        return StandardAlgorithm()

    @abstractmethod
    def value(self, distribution: Distribution) -> float:
        pass


class StandardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        n = distribution.n
        return random.uniform() / sqrt(random.chisquare(n) / n)


class Mean:
    __distribution: Distribution

    @cached_property
    def __value(self) -> float:
        return 0


class Variance:
    __distribution: Distribution

    @cached_property
    def __value(self) -> float:
        n = self.__distribution.n
        if n > 2:
            return n / (n - 2)
        else:
            return inf
