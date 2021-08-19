from statistical_modeling.distributions.binomial import Distribution, Mean, Variance

import unittest
from numpy.testing import assert_equal, assert_allclose

from typing import Final


class TestBinomial(unittest.TestCase):
    d: Final = Distribution(10, 0.5)

    def test_distribution(self):
        assert_equal(self.d.n, 10)
        assert_equal(self.d.p, 0.5)
        assert_equal(self.d.q, 0.5)

    def test_mean(self):
        assert_allclose(
            Mean(self.d),
            5
        )

    def test_variance(self):
        assert_allclose(
            Variance(self.d),
            2.5
        )
