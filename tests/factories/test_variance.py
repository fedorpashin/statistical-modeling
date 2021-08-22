import statistical_modeling as sm

from typing import Final

import unittest


class TestVariance(unittest.TestCase):
    def test_sample(self):
        s: Final = sm.Sample([1, 2, 3])
        self.assertEqual(
            sm.Variance(s),
            sm.SampleVariance(s)
        )

    def test_binomial(self):
        d: Final = sm.binomial.Distribution(10, 0.1)
        self.assertEqual(
            sm.Variance(d),
            sm.binomial.Variance(d)
        )

    def test_chi_square(self):
        d: Final = sm.chi_square.Distribution(10)
        self.assertEqual(
            sm.Variance(d),
            sm.chi_square.Variance(d)
        )

    def test_continuous_uniform(self):
        d: Final = sm.continuous_uniform.Distribution(0, 1)
        self.assertEqual(
            sm.Variance(d),
            sm.continuous_uniform.Variance(d)
        )

    def test_discrete_uniform(self):
        d: Final = sm.discrete_uniform.Distribution(0, 1)
        self.assertEqual(
            sm.Variance(d),
            sm.discrete_uniform.Variance(d)
        )

    def test_exponential(self):
        d: Final = sm.exponential.Distribution(1)
        self.assertEqual(
            sm.Variance(d),
            sm.exponential.Variance(d)
        )

    def test_geometric(self):
        d: Final = sm.geometric.Distribution(0.1)
        self.assertEqual(
            sm.Variance(d),
            sm.geometric.Variance(d)
        )

    def test_logarithmic(self):
        d: Final = sm.logarithmic.Distribution(0.1)
        self.assertEqual(
            sm.Variance(d),
            sm.logarithmic.Variance(d)
        )

    def test_poisson(self):
        d: Final = sm.poisson.Distribution(10)
        self.assertEqual(
            sm.Variance(d),
            sm.poisson.Variance(d)
        )

    def test_standard_normal(self):
        d: Final = sm.standard_normal.Distribution()
        self.assertEqual(
            sm.Variance(d),
            sm.standard_normal.Variance(d)
        )

    def test_student(self):
        d: Final = sm.student.Distribution(10)
        self.assertEqual(
            sm.Variance(d),
            sm.student.Variance(d)
        )
