from abc import ABC, abstractmethod
from final_class import final

from functools import cached_property

__all__ = ['AnySample',
           'Sample',
           'SampleMean',
           'SampleVariance',
           'SampleACF',
           'plot_correlogram',
           'plot_pdf',
           'plot_cdf']


class AnySample(ABC):
    @property
    @abstractmethod
    def x(self) -> list[float]:
        pass

    @cached_property
    def n(self) -> int:
        return len(self.x)


@final
class Sample(AnySample):
    __x: list[float]

    def __init__(self, x: list[float]):
        self.__x = x

    def __str__(self) -> str:
        return str(self.__x)

    @property
    def x(self):
        return self.__x


@final
class SampleMean(float):
    __sample: AnySample

    def __new__(cls, sample: AnySample) -> 'SampleMean':
        return super().__new__(cls, cls.__value(sample))

    def __init__(self, sample: AnySample):
        self.__sample = sample

    @property
    def sample(self):
        return self.__sample

    @staticmethod
    def __value(sample: AnySample) -> float:
        x = sample.x
        n = sample.n
        return sum(x) / n


@final
class SampleVariance(float):
    __sample: AnySample

    def __new__(cls, sample: AnySample) -> 'SampleVariance':
        return super().__new__(cls, cls.__value(sample))

    def __init__(self, sample: AnySample):
        self.__sample = sample

    @property
    def sample(self):
        return self.__sample

    @staticmethod
    def __value(sample: AnySample) -> float:
        x = sample.x
        n = sample.n
        mean = SampleMean(sample)
        return sum([(x_i - mean)**2 for x_i in x]) / n


@final
class SampleACF(float):
    __sample: AnySample
    __f: int

    def __new__(cls, sample: AnySample, f: int) -> 'SampleACF':
        return super().__new__(cls, cls.__value(sample, f))

    def __init__(self, sample: AnySample, f: int):
        self.__sample = sample
        self.__f = f

    @property
    def sample(self):
        return self.__sample

    @property
    def f(self):
        return self.__f

    @staticmethod
    def __value(sample: AnySample, f: int) -> float:
        x = sample.x
        n = sample.n
        mean = SampleMean(sample)
        assert f <= n
        return (sum([(x[i] - mean) * (x[i + f] - mean) for i in range(n - f)])
                / sum([(x[i] - mean)**2 for i in range(n)]))


def plot_correlogram(sample: AnySample, ax) -> None:
    n = sample.n
    ax.plot(range(n), [SampleACF(sample, f) for f in range(n)])


def plot_pdf(sample: AnySample, ax) -> None:
    x = sample.x
    ax.hist(x, bins=10, density=True)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")


def plot_cdf(sample: AnySample, ax) -> None:
    x = sample.x
    ax.hist(x, bins=1000, density=True, cumulative=True)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
