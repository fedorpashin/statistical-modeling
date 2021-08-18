from .base import ContinuousDistribution, ContinuousDistributionAlgorithm, DistributionMean, DistributionVariance
from abc import abstractmethod

from numpy import random


class Distribution(ContinuousDistribution):
    __n: int

    def __init__(self, n: int):
        assert n > 0
        self.__n = n

    @property
    def n(self) -> int:
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
        return sum([random.standard_normal()**2 for _ in range(n)])


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        return n


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        return 2 * n
