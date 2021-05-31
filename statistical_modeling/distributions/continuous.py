from .base import Distribution
from abc import abstractmethod
from enum import Enum

from numpy import random
import math

from functools import cached_property

__all__ = ['ContinuousUniformDistribution',
           'NormalDistribution',
           'ExponentialDistribution',
           'ChiSquareDistribution',
           'StudentDistribution']


class ContinuousDistribution(Distribution):
    class Algorithm(Enum):
        pass

    @abstractmethod
    def value(self, algorithm: Algorithm) -> float:
        pass


class ContinuousUniformDistribution(ContinuousDistribution):
    class Algorithm(Enum):
        standard = 1

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.standard

    _a: int
    _b: int

    def __init__(self, a: int = 0, b: int = 1):
        self._a = a
        self._b = b

    def value(self, algorithm: Algorithm = default_algorithm) -> float:  # type: ignore
        return (self._b - self._a) * random.uniform() + self._a

    @property
    def a(self) -> int:
        return self._a

    @property
    def b(self) -> int:
        return self._b

    @cached_property
    def E(self) -> float:
        return (self._a + self._b) / 2

    @cached_property
    def D(self) -> float:
        return ((self._b - self._a)**2) / 12


class NormalDistribution(ContinuousDistribution):
    class Algorithm(Enum):
        box_miller = 1
        central_limit_theorem = 2

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.box_miller

    _m: float
    _σ: float

    def __init__(self, m: float, σ: float):
        self._m = m
        self._σ = σ

    def value(self, algorithm: Algorithm = default_algorithm) -> float:  # type: ignore
        if algorithm == self.Algorithm.central_limit_theorem:
            return sum([random.uniform() for _ in range(12)]) - 6
        elif algorithm == self.Algorithm.box_miller:
            return math.sqrt(-2 * math.log(random.uniform())) * math.cos(2 * math.pi * random.uniform())

    @property
    def m(self) -> float:
        return self._m

    @property
    def σ(self) -> float:
        return self._σ

    @cached_property
    def E(self) -> float:
        return self._m

    @cached_property
    def D(self) -> float:
        return self._σ**2


class ExponentialDistribution(ContinuousDistribution):
    class Algorithm(Enum):
        standard = 1

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.standard

    _β: float

    def __init__(self, β: float):
        self._β = β

    def value(self, algorithm: Algorithm = default_algorithm) -> float:  # type: ignore
        return -self._β * math.log(random.uniform())

    @property
    def β(self) -> float:
        return self._β

    @cached_property
    def E(self) -> float:
        return self._β

    @cached_property
    def D(self) -> float:
        return self._β**2


class ChiSquareDistribution(ContinuousDistribution):
    class Algorithm(Enum):
        standard = 1

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.standard

    _n: int

    def __init__(self, n: int):
        assert n > 0
        self._n = n

    def value(self, algorithm: Algorithm = default_algorithm) -> float:  # type: ignore
        return sum([random.uniform()**2 for _ in range(self._n)])

    @property
    def n(self) -> int:
        return self._n

    @cached_property
    def E(self) -> float:
        return self._n

    @cached_property
    def D(self) -> float:
        return 2 * self._n


class StudentDistribution(ContinuousDistribution):
    class Algorithm(Enum):
        standard = 1

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.standard

    _n: float

    def __init__(self, n: int):
        assert n > 0
        self._n = n

    def value(self, algorithm: Algorithm = default_algorithm) -> float:  # type: ignore
        return random.uniform() / math.sqrt(random.chisquare(self._n) / self._n)

    @cached_property
    def E(self) -> float:
        return 0

    @cached_property
    def D(self) -> float:
        if self._n > 2:
            return self._n / (self._n - 2)
        else:
            return math.inf
