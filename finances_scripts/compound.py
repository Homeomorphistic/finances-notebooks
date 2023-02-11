# compound.py
"""A module containing base for all computations done in notebooks.

It contains closed-form formulas for compounding interest which is basic idea in finances. It also contains
empirical computations used for testing compounding.

Not to be used by a user.
"""
from operator import mul, truediv
from typing import Union, Iterable

from numpy import array, ndarray, exp, broadcast_arrays


def _compound(d_0: Union[float, ndarray[float]],
              n: Union[int, ndarray[int]],
              p: Union[float, ndarray[float]]
              ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit d_0 after n periods, with periodic interest rate p.

    :param d_0: initial deposit
    :param n: number of periods
    :param p: periodic interest rate
    :return: deposit after n periods

    Matrices have to be broadcastable to obtain any result. Use one of the parameters as a vector or all of them.

    Examples:
    >>> from numpy import array
    >>> _compound(1000, 1, 0.05)
    1050.0
    >>> _compound(array([-1000, 0, 1000]), array([-1, 0, 1]), array([-0.05, 0, 0.05]))
    array([-1052.63157895,     0.        ,  1050.        ])
    """
    return d_0 * (1 + p)**n


def _compound_recursive_helper(d_0: Union[float, ndarray[float]],
                               n: int,
                               p: Union[float, ndarray[float]]
                               ) -> Union[float, ndarray[float]]:
    """Compute recursively compound interest of initial deposit d_0 after n periods, with periodic interest rate p.

    :param d_0: initial deposit
    :param n: number of periods
    :param p: periodic interest rate
    :return: deposit after n periods

    This function is helper for more general _compound_iteratively, use it instead. Matrices have to be broadcastable
    to obtain any result. The number of periods cannot be a vector, because implementing it does not make sense.
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
    """Compute recursively compound interest of initial deposit d_0 after n periods, with periodic interest rate p.

    :param d_0: initial deposit
    :param n: number of periods
    :param p: periodic interest rate
    :return: deposit after n periods

    This function exists solely for testing purposes. Matrices have to be broadcastable to obtain any result. Use one of
    the parameters as a vector or all of them.
    """
    if isinstance(n, ndarray):
        d_0, n, p = broadcast_arrays(d_0, n, p)  # unpack list of arrays
        return array([_compound_recursive_helper(d_0_i, n_i, p_i) for d_0_i, n_i, p_i in zip(d_0, n, p)])
    else:
        return _compound_recursive_helper(d_0, n, p)


def _compound_iterative_helper(d_0: Union[float, Iterable[float]],
                               n: int,
                               p: Union[float, Iterable[float]]
                               ) -> Union[float, Iterable[float]]:
    """Compute iteratively compound interest of initial deposit d_0 after n periods, with periodic interest rate p.

    :param d_0: initial deposit
    :param n: number of periods
    :param p: periodic interest rate
    :return: deposit after n periods

    This function is helper for more general _compound_iteratively, use it instead. Matrices have to be broadcastable
    to obtain any result. The number of periods cannot be a vector, because implementing it does not make sense.
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
    """Compute iteratively compound interest of initial deposit d_0 after n periods, with periodic interest rate p.

    :param d_0: initial deposit
    :param n: number of periods
    :param p: periodic interest rate
    :return: deposit after n periods

    This function exists solely for testing purposes. Matrices have to be broadcastable to obtain any result. Use one of
    the parameters as a vector or all of them.
    """
    if isinstance(n, ndarray):
        d_0, n, p = broadcast_arrays(d_0, n, p)  # unpack list of arrays
        return array([_compound_iterative_helper(d_0_i, n_i, p_i) for d_0_i, n_i, p_i in zip(d_0, n, p)])
    else:
        return _compound_iterative_helper(d_0, n, p)


def _compound_frequently(d_0: Union[float, ndarray[float]],
                         n: Union[int, ndarray[int]],
                         p: Union[float, ndarray[float]],
                         m: Union[int, ndarray[int]]
                         ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit d_0 after n periods, with periodic interest rate p and compounding
    frequency m.

    :param d_0: initial deposit
    :param n: number of periods
    :param p: periodic interest rate
    :param m: compounding frequency
    :return: deposit after n periods

    Matrices have to be broadcastable to obtain any result. Use one of the parameters as a vector or all of them.

    Examples:
    >>> from numpy import array
    >>> _compound_frequently(1000, 2, 0.05, 2)
    1103.8128906249995
    >>> _compound_frequently(array([-1000, 0, 1000]), array([-3, 0, 3]), array([-0.05, 0, 0.05]), array([1, 2, 3]))
    array([-1166.35078   ,     0.        ,  1160.39877496])
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
    """Calculate compound interest of initial deposit d_0 after time t, with periodic interest rate p and continuous
    compounding.

    :param d_0: initial deposit
    :param t: time of deposit
    :param p: periodic interest rate
    :return: deposit after time t

    Matrices have to be broadcastable to obtain any result. Use one of the parameters as a vector or all of them.
    """
    return d_0 * exp(p * t)


if __name__ == '__main__':
    pass
