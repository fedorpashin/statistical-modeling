from statistical_modeling.distributions.exponential import (
    Distribution,
    Mean,
    Variance
)

from typing import Final

import unittest


class TestExponential(unittest.TestCase):
    d: Final = Distribution(1)

    def test_distribution(self):
        self.assertEqual(self.d.β, 1)

    def test_mean(self):
        self.assertAlmostEqual(
            Mean(self.d),
            1
        )

    def test_variance(self):
        self.assertAlmostEqual(
            Variance(self.d),
            1
        )
