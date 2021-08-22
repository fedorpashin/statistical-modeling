from statistical_modeling.distributions import (
    exponential,
    binomial
)
from statistical_modeling import RandomFloat

from typing import Final

import unittest


class TestRandomFloat(unittest.TestCase):
    d: Final = exponential.Distribution(1)

    def test_algorithm(self):
        a: Final = exponential.DefaultAlgorithm(self.d)
        self.assertEqual(
            RandomFloat(self.d, a).algorithm,
            a
        )

    def test_default_algorithm(self):
        self.assertEqual(
            type(RandomFloat(self.d).algorithm),
            type(exponential.DefaultAlgorithm(self.d))
        )

    def test_invalid_combination(self):
        with self.assertRaises(AttributeError):
            RandomFloat(self.d, binomial.CumulativeAlgorithm()) # noqa
