from statistical_modeling.distributions import binomial, exponential
from statistical_modeling import RandomInt

from typing import Final

import unittest


class TestRandomInt(unittest.TestCase):
    d: Final = binomial.Distribution(10, 0.1)

    def test_algorithm(self):
        a: Final = binomial.DefaultAlgorithm(self.d)
        self.assertEqual(
            RandomInt(self.d, a).algorithm,
            a
        )

    def test_default_algorithm(self):
        self.assertEqual(
            type(RandomInt(self.d).algorithm),
            type(binomial.DefaultAlgorithm(self.d))
        )

    def test_invalid_combination(self):
        with self.assertRaises(AttributeError):
            RandomInt(self.d, exponential.StandardAlgorithm()) # noqa
