from unittest import TestCase

from numpy import array


class Test(TestCase):
    def setUp(self) -> None:
        # Future deposit.
        self.d_0 = array([1000.0, 1500, 2000])
        # Number of periods.
        self.n = array([0, 1, 2])
        # Periodic inflation rate.
        self.f = array([.05, .07, .1])
        # Frequency of compounding
        self.m = array([1, 2, 3])

    def test_closed_inflate(self):
        self.fail()
