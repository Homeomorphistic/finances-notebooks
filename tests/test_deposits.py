from unittest import TestCase

from numpy import array
from numpy.testing import assert_allclose

from finances_scripts.deposits import deposit

class Test(TestCase):
    def setUp(self) -> None:
        # Initial deposit.
        self.d_0 = array([1000.0, 1500, 2000])
        # Number of periods.
        self.n = array([0, 1, 2])
        # Periodic interest rate.
        self.p = array([.05, .07, .1])
        # Frequency of compounding
        self.m = array([1, 2, 3])
        # Periodic contributions
        self.c = array([100, 200, 300])
        # Periodic inflation
        self.f = array([.05, .07, .1])
        # Periodic tax
        self.t = array([.18, .19, .2])

    def test_deposit(self):
        # Given the initial deposit, number of periods, periodic interest rate, frequency, periodic contribution,
        # periodic inflation rate, periodic tax
        d_0, n, p, m, c, f, t = 1000, 2, .05, 1, 100, .05, .19
        # When I want to calculate amount of deposit at n-th period
        d_n = deposit(initial_deposit=d_0, periods=n, interest=p, frequency=m, contribution=c, inflation=f, tax=t)
        # Then I get correct answer
        self.assertAlmostEqual(d_n, 1076.36, delta=.01, msg='Deposit amount for closed-form compound is not equal.')
