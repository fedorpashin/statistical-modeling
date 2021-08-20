from .base import DiscreteDistribution, DiscreteDistributionAlgorithm, DistributionMean, DistributionVariance

from math import floor
from numpy import random

from functools import cached_property


class Distribution(DiscreteDistribution):
    __a: int
    __b: int

    def __init__(self, a: int, b: int):
        self.__a = a
        self.__b = b

    @property
    def a(self) -> int:
        return self.__a

    @property
    def b(self) -> int:
        return self.__b

    @cached_property
    def n(self) -> int:
        return self.b - self.a + 1


class Algorithm(DiscreteDistributionAlgorithm[Distribution]):
    @staticmethod
    def default(distribution: Distribution) -> 'Algorithm':
        pass


class StandardAlgorithm(Algorithm):

    def value(self, distribution: Distribution) -> int:
        n = distribution.n
        a = distribution.a
        return floor(n * random.uniform() + a)


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
        n = distribution.n
        return (n**2 - 1) / 12
