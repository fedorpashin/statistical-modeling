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

    def k(self, f: int) -> float:
        assert f < self.n
        return (sum([(self._x[i] - self.E) * (self._x[i + f] - self.E) for i in range(self.n - f)])
                / sum([(self._x[i] - self.E)**2 for i in range(self.n)]))

    @cached_property
    def E(self) -> float:
        return sum(self._x) / self.n

    @cached_property
    def D(self) -> float:
        return sum([(x_i - self.E)**2 for x_i in self._x]) / self.n

    def plot_correlogram(self, ax) -> None:
        ax.plot(range(self.n), [self.k(f) for f in range(self.n)])

    def plot_pdf(self, ax) -> None:
        ax.hist(self._x, bins=10, density=True)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")

    def plot_cdf(self, ax) -> None:
        ax.hist(self._x, bins=1000, density=True, cumulative=True)
        ax.set_xlabel("x")
        ax.set_ylabel("F(x)")
