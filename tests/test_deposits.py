from unittest import TestCase

from numpy import array
from numpy.testing import assert_allclose

from finances_scripts.compound_interest import _compound_frequently
from finances_scripts.inflation import _inflate_frequently
from finances_scripts.deposits import deposit, _deposit_iterate

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
        # Number of periodic contributions
        self.c_n = array([0, 1, 2])
        # Periodic inflation
        self.f = array([.05, .07, .1])
        # Periodic tax
        self.t = array([.18, .19, .2])

    def test_deposit(self):
        # Given the initial deposit, number of periods, periodic interest rate, frequency, periodic contribution,
        # periodic inflation rate, periodic tax
        d_0, n, p, m, c, c_n, f, t = 1000, 2, .05, 1, 100, 1, .05, .19
        # When I want to calculate amount of deposit at n-th period
        d_n = deposit(initial_deposit=d_0, periods=n, interest=p, frequency=m, contribution=c, contribution_periods=c_n,
                      inflation=f, tax=t)
        # Then I get correct answer
        self.assertAlmostEqual(d_n, 1076.36, delta=.01, msg='Deposit amount for closed-form compound is not equal.')

    def test_zero_contrib_deposit(self):
        # Given vectors of initial deposit, number of periods, periodic interest rates, frequencies,
        # zero periodic contributions, periodic inflation rates periodic taxes
        # When I calculate current value of each future deposit
        d_0 = deposit(initial_deposit=self.d_0.copy(), periods=self.n, interest=self.p,
                      frequency=self.m, contribution=0, contribution_periods=0,
                      inflation=self.f, tax=self.t)
        # Then the results match closed-form formula
        desired = _compound_frequently(d_0=self.d_0.copy(), n=self.n, p=self.p*(1-self.t), m=self.m)
        assert_allclose(actual=d_0, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    def test_some_zero_contrib_deposit(self):
        # Given vectors of initial deposit, number of periods, periodic interest rates, frequencies,
        # some zero periodic contributions, periodic inflation rates periodic taxes
        # When I calculate current value of each future deposit
        d_0 = deposit(initial_deposit=self.d_0.copy(), periods=self.n, interest=self.p,
                      frequency=self.m, contribution=array([0, 0, 300]), contribution_periods=array([0, 0, 1]),
                      inflation=self.f, tax=self.t)
        # Then the results match closed-form formula
        desired = array([1000, 1586.256, 2424.82])
        assert_allclose(actual=d_0, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    # def test_iterative_deposit(self):
    #     # Given vectors of initial deposit, number of periods, periodic interest rates, frequencies,
    #     # periodic contributions, periodic inflation rates periodic taxes
    #     # When I calculate current value of each future deposit using iterative method
    #     d_0_iter = _deposit_iterate(initial_deposit=self.d_0.copy(), periods=self.n, interest=self.p,
    #                                 frequency=self.m, contribution=self.c, contribution_periods=self.c_n,
    #                                 inflation=self.f, tax=self.t)
    #     # Then the results match closed-form formula
    #     desired = deposit(initial_deposit=self.d_0.copy(), periods=self.n, interest=self.p,
    #                       frequency=self.m, contribution=self.c, contribution_periods=self.c_n,
    #                       inflation=self.f, tax=self.t)
    #     assert_allclose(actual=d_0_iter, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')
