from statistical_modeling.distributions.geometric import Distribution, Mean, Variance

from typing import Final

import unittest
from numpy.testing import assert_equal, assert_allclose


class TestGeometric(unittest.TestCase):
    d: Final = Distribution(0.5)

    def test_distribution(self):
        assert_equal(self.d.p, 0.5)
        assert_equal(self.d.q, 0.5)

    def test_mean(self):
        assert_allclose(
            Mean(self.d),
            2
        )

    def test_variance(self):
        assert_allclose(
            Variance(self.d),
            2
        )
