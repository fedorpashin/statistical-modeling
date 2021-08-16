from functools import cached_property

__all__ = ['Sample']


class Sample:
    _x: list[float]

    def __init__(self, x: list[float]):
        self._x = x

    def __str__(self) -> str:
        return str(self._x)

    @property
    def x(self):
        return self._x

    @cached_property
    def n(self):
        return len(self._x)


class Mean:
    _sample: Sample

    def __init__(self, sample: Sample):
        self._sample = sample

    @cached_property
    def _value(self) -> float:
        x = self._sample.x
        n = self._sample.n
        return sum(x) / n


class Variance:
    _sample: Sample

    def __init__(self, sample: Sample):
        self._sample = sample

    @cached_property
    def _value(self) -> float:
        x = self._sample.x
        n = self._sample.n
        mean = Mean(self._sample)._value

        return sum([(x_i - mean)**2 for x_i in x]) / n


class ACF:
    _sample: Sample
    _f: int

    def __init__(self, sample: Sample, f: int):
        self._sample = sample
        self._f = f

    @cached_property
    def _value(self) -> float:
        x = self._sample.x
        n = self._sample.n
        f = self._f
        mean = Mean(self._sample)._value
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
