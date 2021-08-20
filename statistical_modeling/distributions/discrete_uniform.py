from .base import DiscreteDistribution, DiscreteDistributionAlgorithm, DistributionMean, DistributionVariance
from dataclasses import dataclass
from final_class import final
from overrides import overrides

from math import floor
from numpy import random

from functools import cached_property


@final
@dataclass(frozen=True)
class Distribution(DiscreteDistribution):
    a: int
    b: int

    def __post__init__(self):
        assert self.a < self.b

    @cached_property
    def n(self) -> int:
        return self.b - self.a + 1


class Algorithm(DiscreteDistributionAlgorithm[Distribution]):
    pass


@final
class DefaultAlgorithm:
    def __new__(cls, distribution: Distribution) -> Algorithm:
        pass


@final
class StandardAlgorithm(Algorithm):
    @overrides
    def value(self, distribution: Distribution) -> int:
        n = distribution.n
        a = distribution.a
        return floor(n * random.uniform() + a)


@final
class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        a = distribution.a
        b = distribution.b
        return (a + b) / 2


@final
class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        return (n**2 - 1) / 12
