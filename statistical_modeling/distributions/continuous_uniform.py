from .base import ContinuousDistribution, ContinuousDistributionAlgorithm
from abc import abstractmethod

from numpy import random

from functools import cached_property


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


class Algorithm(ContinuousDistributionAlgorithm['Algorithm']):
    @abstractmethod
    def value(self, distribution: Distribution) -> float:
        pass

    @classmethod
    def default(cls, distribution: Distribution) -> 'Algorithm':
        return StandardAlgorithm()


class StandardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        a = distribution.a
        b = distribution.b
        return (b - a) * random.uniform() + a


class Mean:
    __distribution: Distribution

    @cached_property
    def __value(self) -> float:
        a = self.__distribution.a
        b = self.__distribution.b
        return (a + b) / 2


class Variance:
    __distribution: Distribution

    @cached_property
    def __value(self) -> float:
        a = self.__distribution.a
        b = self.__distribution.b
        return ((b - a)**2) / 12
