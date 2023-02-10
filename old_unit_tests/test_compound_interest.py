from unittest import TestCase

from numpy import array
from numpy.testing import assert_allclose

from finances_scripts.compound_interest import _compound, _compound_iterate, _compound_recursive_helper, _compound_frequently


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

    def test_closed_compound(self):
        # Given the initial deposit, number of periods and periodic interest rate
        # When I want to calculate amount of deposit at n-th period
        d_n = _compound(d_0=self.d_0[0], n=self.n[-1], p=self.p[0])
        # Then I get correct answer
        self.assertAlmostEqual(d_n, 1102.5, delta=.01, msg='Deposit amount for closed-form compound is not equal.')

    def test_iterative_and_recursive_compound(self):
        # Given the initial deposit, number of periods and periodic interest rate
        # When I calculate amount of deposit at n-th period using iterative and recursive methods
        d_n = _compound(d_0=self.d_0[0], n=self.n[-1], p=self.p[0])
        d_n_iter = _compound_iterate(d_0=self.d_0[0], n=self.n[-1], p=self.p[0])
        d_n_rec = _compound_recursive_helper(d_0=self.d_0[0], n=self.n[-1], p=self.p[0])
        # Then I get equal results
        self.assertAlmostEqual(d_n, d_n_iter, delta=.01, msg='Deposit amounts for closed and iterative compound'
                                                             ' are not equal.')
        self.assertAlmostEqual(d_n, d_n_rec, delta=.01, msg='Deposit amounts for closed and recursive compound'
                                                            ' are not equal.')

    def test_deposit_vector_compound(self):
        # Given the initial deposits, vector of number of periods and periodic interest rate
        # When I calculate amount of deposit in a few periods
        d_n = _compound(d_0=self.d_0, n=self.n[-1], p=self.p[0])
        # Then I get a different result for each deposit
        desired = [1102.5, 1653.75, 2205]
        assert_allclose(actual=d_n, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    def test_period_vector_compound(self):
        # Given the initial deposit, vector of number of periods and periodic interest rate
        # When I calculate amount of deposit for a few periods
        d_n = _compound(d_0=self.d_0[0], n=self.n, p=self.p[0])
        # Then I get a different result for each period
        desired = [1000, 1050, 1102.5]
        assert_allclose(actual=d_n, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    def test_interest_vector_compound(self):
        # Given the initial deposit, number of periods and vector of periodic interest rates
        # When I calculate amount of deposit for a few interest rates
        d_n = _compound(d_0=self.d_0[0], n=self.n[-1], p=self.p)
        # Then I get a different result for each interest rate
        desired = [1102.5, 1144.9, 1210]
        assert_allclose(actual=d_n, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    def test_all_vectors_compound(self):
        # Given vectors of initial deposits, number of periods and periodic interest rates with the same shape
        # When I calculate amount of deposit for each triplet
        d_n = _compound(d_0=self.d_0, n=self.n, p=self.p)
        # Then I get a result for each triplet
        desired = [1000, 1605, 2420]
        assert_allclose(actual=d_n, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    def test_non_broadcastable_compound(self):
        # Given vectors of initial deposits, number of periods and periodic interest rates with different shapes
        # When I calculate amount of deposit for each triplet
        # Then I get an error with broadcasting
        self.assertRaises(ValueError, lambda: _compound(d_0=self.d_0, n=self.n[:2], p=self.p))

    def test_all_vectors_iterative_compound(self):
        # Given vectors of initial deposits, number of periods and periodic interest rates with different shapes
        # When I calculate amount of deposit for each triplet using iterative method
        d_n_iter = _compound_iterate(d_0=self.d_0.copy(), n=self.n, p=self.p)
        # Then it has the equal results as in closed-form formula
        desired = _compound(d_0=self.d_0.copy(), n=self.n, p=self.p)
        assert_allclose(actual=d_n_iter, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    def test_frequent_compound(self):
        # Given initial deposit, number of periods, periodic interest rate and frequency
        # When I calculate amount of deposit
        d_n = _compound_frequently(d_0=1000, n=1, p=.05, m=2)
        # Then I get correct result
        self.assertAlmostEqual(d_n, 1050.625, delta=.01, msg=f'Deposit with frequency {2} doesn\'t match')

    def test_frequency_vector_compound(self):
        # Given the initial deposit, number of periods, periodic interest rates and vector of frequencies
        # When I calculate amount of deposit for a few frequencies
        d_n = _compound_frequently(d_0=1000, n=1, p=.05, m=self.m)
        # Then I get a different result for each frequency
        desired = [1050, 1050.625, 1050.837956072]
        assert_allclose(actual=d_n, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')

    def test_zero_frequency_compound_fails(self):
        # Given initial deposit, number of periods, periodic interest rate and frequency equal to zero
        m = 0
        # When I calculate amount of deposit
        # Then I get an error, because of division by zero
        self.assertRaises(AssertionError, lambda: _compound_frequently(d_0=1000, n=1, p=.05, m=m))

    def test_zero_frequency_vector_compound_fails(self):
        # Given initial deposit, number of periods, periodic interest rate and vector of frequencies with some
        # equal to zero
        m = array([-1, 0, 5])
        # When I calculate amount of deposit
        # Then I get an error, because of division by zero
        self.assertRaises(AssertionError, lambda: _compound_frequently(d_0=1000, n=1, p=.05, m=m))

    def test_all_vectors_compound_frequently(self):
        # Given vectors of initial deposits, number of periods, periodic interest rates and frequencies
        # When I calculate amount of deposit for each
        d_n = _compound_frequently(d_0=self.d_0.copy(), n=self.n, p=self.p, m=self.m)
        # Then it has the equal results as in iterative method
        desired = _compound_iterate(d_0=self.d_0.copy(), n=self.n*self.m, p=self.p/self.m)
        assert_allclose(actual=d_n, desired=desired, atol=.01, err_msg='Some deposit amounts do not match.')
