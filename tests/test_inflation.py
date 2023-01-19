from unittest import TestCase

from numpy import array
from numpy.testing import assert_allclose

from finances_scripts.inflation import _inflate, _inflate_iterate, _inflate_frequently


class Test(TestCase):
    def setUp(self) -> None:
        # Future deposit.
        self.d_n = array([1000.0, 1500, 2000])
        # Number of periods.
        self.n = array([0, 1, 2])
        # Periodic inflation rate.
        self.f = array([.05, .07, .1])
        # Frequency of compounding
        self.m = array([1, 2, 3])

    def test_closed_inflate(self):
        # Given future deposit, number of periods and periodic inflation rate
        d_n, n, f = 1000, 2, .05
        # When I calculate current value of future deposit after n periods
        d_0 = _inflate(d_n=d_n, n=n, f=f)
        # Then I get correct result
        self.assertAlmostEqual(d_0, 907.03, delta=.01, msg='Deposit amount for closed-form compound is not equal.')

    def test_iterative_inflation(self):
        # Given vectors future deposit, number of periods and periodic inflation rate
        # When I calculate current value of each future deposit using iterative method
        d_0_iter = _inflate_iterate(d_n=self.d_n.copy(), n=self.n, f=self.f)
        # Then the results match closed-form formula
        desired = _inflate(d_n=self.d_n.copy(), n=self.n, f=self.f)
        assert_allclose(actual=d_0_iter, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    def test_frequent_inflation(self):
        # Given vectors future deposit, number of periods, periodic inflation rate and frequencies
        # When I calculate current value of each future deposit
        d_0 = _inflate_frequently(d_n=self.d_n.copy(), n=self.n, f=self.f, m=self.m)
        # Then the result match iterative method
        desired = _inflate_iterate(d_n=self.d_n.copy(), n=self.n*self.m, f=self.f/self.m)
        assert_allclose(actual=d_0, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')
