# inflation.py
"""A module containing base for inflation.

It contains closed-form formulas for basic ideas in finances and also their empirical computations used for testing.

Not to be used by a user.
"""
from typing import Union, Iterable

from numpy import ndarray

from .compound_interest import _compound, _compound_frequently, _compound_continuous


def _inflate(d_n: Union[float, ndarray[float]],
             n: Union[int, ndarray[int]],
             f: Union[float, ndarray[float]]
             ) -> Union[float, ndarray[float]]:
    """Calculate current value of future money (in current money terms) after n periods with inflation rate f.

    Matrices have to be broadcastable to obtain any result.

    :param d_n: future deposit or matrix of future deposits
    :param n: number of periods or matrix of numbers
    :param f: periodic inflation rate or matrix of inflation rates
    :return: current value after n periods or matrix of current values
    """
    return _compound(d_n, -n, f)


def _inflate_iterate(d_n: Union[float, Iterable[float]],
                     n: Union[float, Iterable[float]],
                     f: Union[float, Iterable[float]]
                     ) -> Union[float, Iterable[float]]:
    """Calculate current value of future money (in current money terms) after n periods with inflation rate f
    using iterative method.

    Matrices have to be broadcastable to obtain any result.

    :param d_n: future deposit or matrix of future deposits
    :param n: number of periods or matrix of numbers
    :param f: periodic inflation rate or matrix of inflation rates
    :return: current value after n periods or matrix of current values
    """
    # Convert floats to list of one float for iteration.
    d_n = d_n if isinstance(d_n, Iterable) else [d_n]
    n = n if isinstance(n, Iterable) else [n]
    f = f if isinstance(f, Iterable) else [f]

    for i, (d_i, n_i, f_i) in enumerate(zip(d_n, n, f)):
        d_0 = d_i
        for _ in range(n_i):
            d_0 /= (1 + f_i)
        d_n[i] = d_0
    return d_n


def _inflate_frequently(d_n: Union[float, ndarray[float]],
                        n: Union[int, ndarray[int]],
                        f: Union[float, ndarray[float]],
                        m: Union[int, ndarray[int]]
                        ) -> Union[float, ndarray[float]]:
    """Calculate current value of future money (in current money terms) after n periods with inflation rate f
    and compounding frequency m.

    Matrices have to be broadcastable to obtain any result.

    :param d_n: future deposit or matrix of future deposits
    :param n: number of periods or matrix of numbers
    :param f: periodic inflation rate or matrix of inflation rates
    :param m: compounding frequency or matrix of frequencies
    :return: current value after n periods or matrix of current values
    """
    return _compound_frequently(d_n, -n, f, m)


def _inflate_continuous(d_n: Union[float, ndarray[float]],
                        t: Union[float, ndarray[float]],
                        f: Union[float, ndarray[float]]
                        ) -> Union[float, ndarray[float]]:
    """Calculate current value of future money (in current money terms) after n periods with inflation rate f.

    Matrices have to be broadcastable to obtain any result.

    :param d_n: future deposit or matrix of future deposits
    :param t: time or matrix of times
    :param f: periodic inflation rate or matrix of inflation rates
    :return: current value after n periods or matrix of current values
    """
    return _compound_continuous(d_n, -t, f)