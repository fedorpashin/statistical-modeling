from functools import cached_property

__all__ = ['Sample',
           'SampleMean',
           'SampleVariance',
           'ACF']


class Sample:
    __x: list[float]

    def __init__(self, x: list[float]):
        self.__x = x

    def __str__(self) -> str:
        return str(self.__x)

    @property
    def x(self):
        return self.__x

    @cached_property
    def n(self):
        return len(self.__x)


class SampleMean(float):
    __sample: Sample

    def __new__(cls, sample: Sample) -> 'SampleMean':
        return super().__new__(cls, cls.__value(sample))

    def __init__(self, sample: Sample):
        self.__sample = sample

    @property
    def sample(self):
        return self.__sample

    @staticmethod
    def __value(sample: Sample) -> float:
        x = sample.x
        n = sample.n
        return sum(x) / n


class SampleVariance(float):
    __sample: Sample

    def __new__(cls, sample: Sample) -> 'SampleVariance':
        return super().__new__(cls, cls.__value(sample))

    def __init__(self, sample: Sample):
        self.__sample = sample

    @property
    def sample(self):
        return self.__sample

    @staticmethod
    def __value(sample: Sample) -> float:
        x = sample.x
        n = sample.n
        mean = SampleMean(sample)
        return sum([(x_i - mean)**2 for x_i in x]) / n


class ACF(float):
    __sample: Sample
    __f: int

    def __new__(cls, sample: Sample, f: int) -> 'ACF':
        return super().__new__(cls, cls.__value(sample, f))

    def __init__(self, sample: Sample, f: int):
        self.__sample = sample
        self.__f = f

    @property
    def sample(self):
        return self.__sample

    @property
    def f(self):
        return self.__f

    @staticmethod
    def __value(sample: Sample, f: int) -> float:
        x = sample.x
        n = sample.n
        mean = SampleMean(sample)
        assert f < n
        return (sum([(x[i] - mean) * (x[i + f] - mean) for i in range(n - f)])
                / sum([(x[i] - mean)**2 for i in range(n)]))


def plot_correlogram(sample: Sample, ax) -> None:
    n = sample.n
    ax.plot(range(n), [ACF(sample, f) for f in range(n)])


def plot_pdf(sample: Sample, ax) -> None:
    x = sample.x
    ax.hist(x, bins=10, density=True)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")


def plot_cdf(sample: Sample, ax) -> None:
    x = sample.x
    ax.hist(x, bins=1000, density=True, cumulative=True)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
