from .base import ContinuousDistribution, ContinuousDistributionAlgorithm, DistributionMean, DistributionVariance
from abc import abstractmethod

from numpy import random


class Distribution(ContinuousDistribution):
    __a: int
    __b: int

    def __init__(self, a: int = 0, b: int = 1):
        self.__a = a
        self.__b = b

    @property
    def a(self) -> int:
        return self.__a

    @property
    def b(self) -> int:
        return self.__b


class Algorithm(ContinuousDistributionAlgorithm[Distribution]):
    @abstractmethod
    def value(self, distribution: Distribution) -> float:
        pass

    @staticmethod
    def default(distribution: Distribution) -> 'Algorithm':
        return StandardAlgorithm()


class StandardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        a = distribution.a
        b = distribution.b
        return (b - a) * random.uniform() + a


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        a = distribution.a
        b = distribution.b
        return (a + b) / 2


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        a = distribution.a
        b = distribution.b
        return ((b - a)**2) / 12
