from functools import cached_property

__all__ = ['Sample']


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


class Mean:
    __sample: Sample

    def __init__(self, sample: Sample):
        self.__sample = sample

    @cached_property
    def __value(self) -> float:
        x = self.__sample.x
        n = self.__sample.n
        return sum(x) / n


class Variance:
    __sample: Sample

    def __init__(self, sample: Sample):
        self.__sample = sample

    @cached_property
    def __value(self) -> float:
        x = self.__sample.x
        n = self.__sample.n
        mean = Mean(self.__sample).__value

        return sum([(x_i - mean)**2 for x_i in x]) / n


class ACF:
    __sample: Sample
    __f: int

    def __init__(self, sample: Sample, f: int):
        self.__sample = sample
        self.__f = f

    @cached_property
    def __value(self) -> float:
        x = self.__sample.x
        n = self.__sample.n
        f = self.__f
        mean = Mean(self.__sample).__value
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
