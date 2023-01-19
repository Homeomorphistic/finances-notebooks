# compound_interest.py
"""A module containing base for whole computations done in notebooks.

It contains closed-form formulas for basic ideas in finances and also their empirical computations used for testing.

Not to be used by a user.
"""

from typing import Union, Iterable

from numpy import ndarray, exp


def _compound(d_0: Union[float, ndarray[float]],
              n: Union[int, ndarray[int]],
              p: Union[float, ndarray[float]]
              ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit d_0 after n periods.

    Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or matrix of initial deposits
    :param n: number of periods or matrix of numbers
    :param p: periodic interest rate or matrix of interests
    :return: deposit after n periods or matrix of deposits
    """
    return d_0 * (1 + p)**n


def _compound_recursive(d_0: float, n: int, p: float) -> float:
    """Compute compound interest of initial deposit d_0 after n periods recursively.

    This method exists solely for testing purposes.

        :param d_0: initial deposit
        :param n: number of periods
        :param p: periodic interest rate
        :return: deposit after n periods
        """
    assert n >= 0, 'Number of periods has to be non-negative integer (using recursive method).'
    if n == 0:
        return d_0
    else:
        return _compound_recursive(d_0, n-1, p) * (1+p)


def _compound_iterate(d_0: Union[float, Iterable[float]],
                      n: Union[int, Iterable[int]],
                      p: Union[float, Iterable[float]]
                      ) -> Union[float, Iterable[float]]:
    """Compute compound interest of initial deposit d_0 after n periods iteratively.

    This method exists solely for testing purpose. Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or iterable of initial deposits
    :param n: number of periods or iterable of numbers
    :param p: periodic interest rate or iterable of interests
    :return: deposit after n periods or iterable of deposits
    """
    # Convert floats to list of one float for iteration.
    d_0 = d_0 if isinstance(d_0, Iterable) else [d_0]
    n = n if isinstance(n, Iterable) else [n]
    p = p if isinstance(p, Iterable) else [p]

    for i, (d_i, n_i, p_i) in enumerate(zip(d_0, n, p)):
        d_n = d_i
        for _ in range(n_i):
            d_n *= (1+p_i)
        d_0[i] = d_n
    return d_0


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


def contribute_compound(d_0: float, n: int, p: float, c: float):
    """Calculate compound interest of initial deposit d_0 after n periods with contributions c per period.

    :param d_0: initial deposit
    :param n: number of periods
    :param p: periodic interest rate
    :param c: contribution to the deposit per period
    :return: deposit after n periods
    """
    return _compound(d_0, n, p) + c * (1 - (1 + p) ** n) / (-p)


if __name__ == '__main__':
    pass
