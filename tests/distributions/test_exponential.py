from statistical_modeling.distributions.exponential import Distribution, Mean, Variance

from typing import Final

import unittest
from numpy.testing import assert_equal, assert_allclose


class TestExponential(unittest.TestCase):
    d: Final = Distribution(1)

    def test_distribution(self):
        assert_equal(self.d.β, 1)

    def test_mean(self):
        assert_allclose(
            Mean(self.d),
            1
        )

    def test_variance(self):
        assert_allclose(
            Variance(self.d),
            1
        )
