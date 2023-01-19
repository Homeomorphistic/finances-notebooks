# contributions.py
"""A module containing formulas for compounding with contributing.

It contains closed-form formulas for basic ideas in finances and also their empirical computations used for testing.

Not to be used by a user.
"""
from typing import Union, Iterable

from numpy import ndarray, exp

from .compound_interest import _compound

def _compound_and_contribute(d_0: Union[float, ndarray[float]],
                             n: Union[int, ndarray[int]],
                             p: Union[float, ndarray[float]],
                             c: Union[float, ndarray[float]]
                             ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit d_0 after n periods with contributions c.

    Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or matrix of initial deposits
    :param n: number of periods or matrix of numbers
    :param p: periodic interest rate or matrix of interests
    :param c: periodic contribution or matrix of contributions
    :return: deposit after n periods or matrix of deposits
    """
    return _compound(d_0, n, p) + c * (1 - (1+p)**n) / (-p) - c  # -c -> without last payment


def _compound_and_contribute_iterate(d_0: Union[float, ndarray[float]],
                                     n: Union[int, ndarray[int]],
                                     p: Union[float, ndarray[float]],
                                     c: Union[float, ndarray[float]]
                                     ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit d_0 after n periods with contributions c using iterative method.

    Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or matrix of initial deposits
    :param n: number of periods or matrix of numbers
    :param p: periodic interest rate or matrix of interests
    :param c: periodic contribution or matrix of contributions
    :return: deposit after n periods or matrix of deposits
    """
    # Convert floats to list of one float for iteration.
    d_0 = d_0 if isinstance(d_0, Iterable) else [d_0]
    n = n if isinstance(n, Iterable) else [n]
    p = p if isinstance(p, Iterable) else [p]
    c = c if isinstance(c, Iterable) else [c]

    for i, (d_i, n_i, p_i, c_i) in enumerate(zip(d_0, n, p, c)):
        d_n = d_i
        for _ in range(n_i):
            d_n = d_n * (1 + p_i) + c_i
        d_0[i] = d_n
    return d_0 - c  # -c -> without last payment


def _compound_and_contribute_frequently(d_0: Union[float, ndarray[float]],
                                        n: Union[int, ndarray[int]],
                                        p: Union[float, ndarray[float]],
                                        m: Union[int, ndarray[int]],
                                        c: Union[float, ndarray[float]]
                                        ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit d_0 after n periods with contributions c.

    Matrices have to be broadcastable to obtain any result.

    :param d_0: initial deposit or matrix of initial deposits
    :param n: number of periods or matrix of numbers
    :param p: periodic interest rate or matrix of interests
    :param m: compounding frequency or matrix of frequencies
    :param c: periodic contribution or matrix of contributions
    :return: deposit after n periods or matrix of deposits
    """
    return _compound_and_contribute(d_0, n*m, p/m, c/m)
