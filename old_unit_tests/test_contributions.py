from unittest import TestCase

from numpy import array
from numpy.testing import assert_allclose

from finances_scripts.contributions import _compound_and_contribute, _compound_and_contribute_iterate, _compound_and_contribute_frequently

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

    def test_compound_and_contribute(self):
        # Given the initial deposit, number of periods, periodic interest rate and periodic contribution
        d_0, n, p, c = 1000, 2, .05, 100
        # When I want to calculate amount of deposit at n-th period
        d_n = _compound_and_contribute(d_0=d_0, n=n, p=p, c=c)
        # Then I get correct answer
        self.assertAlmostEqual(d_n, 1207.5, delta=.01, msg='Deposit amount for closed-form compound is not equal.')

    def test_iterative_compound_and_contribute(self):
        # Given vectors future deposit, number of periods, periodic inflation rate and periodic contributions
        # When I calculate current value of each future deposit using iterative method
        d_0_iter = _compound_and_contribute_iterate(d_0=self.d_0.copy(), n=self.n, p=self.p, c=self.c)
        # Then the results match closed-form formula
        desired = _compound_and_contribute(d_0=self.d_0.copy(), n=self.n, p=self.p, c=self.c)
        assert_allclose(actual=d_0_iter, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    def test_frequent_compound_and_contribute(self):
        # Given vectors future deposit, number of periods, periodic inflation rate and periodic contributions
        # When I calculate current value of each future deposit
        d_0 = _compound_and_contribute_frequently(d_0=self.d_0.copy(), n=self.n, p=self.p, m=self.m, c=self.c)
        # Then the result match iterative method
        desired = _compound_and_contribute_iterate(d_0=self.d_0.copy(), n=self.n*self.m, p=self.p/self.m, c=self.c/self.m)
        assert_allclose(actual=d_0, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')
