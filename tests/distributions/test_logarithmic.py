from statistical_modeling.distributions.logarithmic import (
    Distribution,
    Mean,
    Variance
)

from typing import Final

import unittest


class TestLogarithmic(unittest.TestCase):
    d: Final = Distribution(0.5)

    def test_distribution(self):
        self.assertEqual(self.d.q, 0.5)
        self.assertEqual(self.d.p, 0.5)

    def test_mean(self):
        self.assertAlmostEqual(
            Mean(self.d),
            1.44270,
            places=5
        )

    def test_variance(self):
        self.assertAlmostEqual(
            Variance(self.d),
            0.80402,
            places=5
        )
