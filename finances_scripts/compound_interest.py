# compound_interest.py
"""A module containing base for all computations done in notebooks.

It contains closed-form formulas for basic ideas in finances and also their empirical computations used for testing.

Not to be used by a user.
"""
from operator import mul, truediv
from typing import Union, Iterable

import numpy as np
from numpy import array, ndarray, exp


def _compound(d_0: Union[float, ndarray[float]],
              n: Union[int, ndarray[int]],
              p: Union[float, ndarray[float]]
              ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit d_0 after n periods.

    Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or matrix of initial deposits
    :param n: number of periods or matrix of periods
    :param p: periodic interest rate or matrix of interests
    :return: deposit after n periods or matrix of deposits
    """
    return d_0 * (1 + p)**n


def _compound_recursive_helper(d_0: Union[float, ndarray[float]],
                               n: int,
                               p: Union[float, ndarray[float]]
                               ) -> Union[float, ndarray[float]]:
    """Compute compound interest of initial deposit d_0 after n periods recursively.

    This method exists solely for testing purposes. Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or matrix of initial deposits
    :param n: number of periods
    :param p: periodic interest rate or matrix of interests
    :return: deposit after n periods or matrix of deposits
    """
    op = mul if n >= 0 else truediv  # if n is negative then we need to divide.
    if n == 0:
        return d_0
    else:
        n = n-1 if n > 0 else n+1  # if n is negative then we need to increase it to reach 0.
        return op(_compound_recursive_helper(d_0, n, p), 1 + p)


def _compound_recursive(d_0: Union[float, ndarray[float]],
                        n: Union[int, ndarray[int]],
                        p: Union[float, ndarray[float]]
                        ) -> Union[float, ndarray[float]]:
    """Compute compound interest of initial deposit d_0 after n periods recursively.

    This method exists solely for testing purposes. Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or matrix of initial deposits
    :param n: number of periods or matrix of periods
    :param p: periodic interest rate or matrix of interests
    :return: deposit after n periods or matrix of deposits
    """
    if isinstance(n, ndarray):
        d_0, n, p = np.broadcast_arrays(d_0, n, p)  # unpack list of arrays
        return array([_compound_recursive_helper(d_0_i, n_i, p_i) for d_0_i, n_i, p_i in zip(d_0, n, p)])
    else:
        return _compound_recursive_helper(d_0, n, p)


def _compound_iterative_helper(d_0: Union[float, Iterable[float]],
                               n: int,
                               p: Union[float, Iterable[float]]
                               ) -> Union[float, Iterable[float]]:
    """Compute compound interest of initial deposit d_0 after n periods iteratively.

    This method exists solely for testing purpose. Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or iterable of initial deposits
    :param n: number of periods
    :param p: periodic interest rate or iterable of interests
    :return: deposit after n periods or iterable of deposits
    """
    op = mul if n >= 0 else truediv  # if n is negative then we need to divide.
    n = abs(n)
    d_n = d_0
    for i in range(n):
        d_n = op(d_n, 1+p)
    return d_n


def _compound_iterative(d_0: Union[float, Iterable[float]],
                        n: Union[int, ndarray[int]],
                        p: Union[float, Iterable[float]]
                        ) -> Union[float, Iterable[float]]:
    """Compute compound interest of initial deposit d_0 after n periods iteratively.

    This method exists solely for testing purpose. Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or iterable of initial deposits
    :param n: number of periods
    :param p: periodic interest rate or iterable of interests
    :return: deposit after n periods or iterable of deposits
    """
    if isinstance(n, ndarray):
        d_0, n, p = np.broadcast_arrays(d_0, n, p)  # unpack list of arrays
        return array([_compound_iterative_helper(d_0_i, n_i, p_i) for d_0_i, n_i, p_i in zip(d_0, n, p)])
    else:
        return _compound_iterative_helper(d_0, n, p)


def _compound_frequently(d_0: Union[float, ndarray[float]],
                         n: Union[int, ndarray[int]],
                         p: Union[float, ndarray[float]],
                         m: Union[int, ndarray[int]]
                         ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit d_0 after n periods with compounding frequency m.

    Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or matrix of initial deposits
    :param n: number of periods or matrix of numbers
    :param p: periodic interest rate or matrix of interests
    :param m: compounding frequency or matrix of frequencies
    :return: deposit after n periods or matrix of deposits
    """
    if isinstance(m, ndarray):
        assert (m > 0).all(), 'Frequencies have to be a non-negative integers.'
    else:
        assert m > 0, 'Frequency has to be a non-negative integer.'

    return _compound(d_0, n * m, p / m)


def _compound_continuous(d_0: Union[float, ndarray[float]],
                         t: Union[float, ndarray[float]],
                         p: Union[float, ndarray[float]]
                         ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit d_0 after t time with continuous compounding.

    Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or matrix of initial deposits
    :param t: time or matrix of times
    :param p: periodic interest rate or matrix of interests
    :return: deposit after t time or matrix of deposits
        """
    return d_0 * exp(p * t)



if __name__ == '__main__':
    pass
