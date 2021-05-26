from statistical_modeling.distributions.discrete import *

import unittest
from numpy.testing import assert_allclose


class TestDiscreteUniform(unittest.TestCase):
    def test_quantities(self):
        d = DiscreteUniformDistribution(1, 100)
        with self.subTest():
            assert_allclose(d.E, 50.5)
        with self.subTest():
            assert_allclose(d.D, 833.25)


class TestBinomial(unittest.TestCase):
    def test_quantities(self):
        d = BinomialDistribution(10, 0.5)
        with self.subTest():
            assert_allclose(d.E, 5)
        with self.subTest():
            assert_allclose(d.D, 2.5)


class TestGeometric(unittest.TestCase):
    def test_quantities(self):
        d = GeometricDistribution(0.5)
        with self.subTest():
            assert_allclose(d.E, 2)
        with self.subTest():
            assert_allclose(d.D, 2)


class TestPoisson(unittest.TestCase):
    def test_quantities(self):
        d = PoissonDistribution(10)
        with self.subTest():
            assert_allclose(d.E, 10)
        with self.subTest():
            assert_allclose(d.D, 10)


class TestLogarithmic(unittest.TestCase):
    def test_quantities(self):
        d = LogarithmicDistribution(0.5)
        with self.subTest():
            assert_allclose(d.E, 1.44270, rtol=1e-5)
        with self.subTest():
            assert_allclose(d.D, 0.80402, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
