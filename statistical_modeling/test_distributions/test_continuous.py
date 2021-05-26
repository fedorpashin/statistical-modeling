from statistical_modeling.distributions.continuous import *

import unittest
from numpy.testing import assert_allclose


class TestContinuousUniform(unittest.TestCase):
    def test_quantities(self):
        d = ContinuousUniformDistribution(0, 1)
        with self.subTest():
            assert_allclose(d.E, 0.5)
        with self.subTest():
            assert_allclose(d.D, 0.08333, rtol=1e-04)


class TestNormal(unittest.TestCase):
    def test_quantities(self):
        d = NormalDistribution(0, 1)
        with self.subTest():
            assert_allclose(d.E, 0)
        with self.subTest():
            assert_allclose(d.D, 1)


class TestExponential(unittest.TestCase):
    def test_quantities(self):
        d = ExponentialDistribution(1)
        with self.subTest():
            assert_allclose(d.E, 1)
        with self.subTest():
            assert_allclose(d.D, 1)


class TestChiSquare(unittest.TestCase):
    def test_quantities(self):
        d = ChiSquareDistribution(10)
        with self.subTest():
            assert_allclose(d.E, 10)
        with self.subTest():
            assert_allclose(d.D, 20)


class TestStudent(unittest.TestCase):
    def test_quantities(self):
        d = StudentDistribution(10)
        with self.subTest():
            assert_allclose(d.E, 0)
        with self.subTest():
            assert_allclose(d.D, 1.25)


if __name__ == '__main__':
    unittest.main()
