from statistical_modeling.distributions.geometric import (
    Distribution,
    Mean,
    Variance
)

from typing import Final

import unittest


class TestGeometric(unittest.TestCase):
    d: Final = Distribution(0.5)

    def test_distribution(self):
        self.assertEqual(self.d.p, 0.5)
        self.assertEqual(self.d.q, 0.5)

    def test_mean(self):
        self.assertAlmostEqual(
            Mean(self.d),
            2
        )

    def test_variance(self):
        self.assertAlmostEqual(
            Variance(self.d),
            2
        )
