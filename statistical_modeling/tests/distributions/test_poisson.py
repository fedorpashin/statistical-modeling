from statistical_modeling.distributions.poisson import Distribution, Mean, Variance

from typing import Final

import unittest
from numpy.testing import assert_equal, assert_allclose


class TestPoisson(unittest.TestCase):
    d: Final = Distribution(10)

    def test_distribution(self):
        assert_equal(self.d.µ, 10)

    def test_mean(self):
        assert_allclose(
            Mean(self.d),
            10
        )

    def test_variance(self):
        assert_allclose(
            Variance(self.d),
            10
        )
