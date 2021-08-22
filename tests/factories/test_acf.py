import statistical_modeling as sm

from typing import Final

import unittest


class TestACF(unittest.TestCase):
    def test(self):
        s: Final = sm.Sample([1, 2, 3])
        f: Final = 3
        self.assertEqual(sm.ACF(s, f), sm.SampleACF(s, f))
