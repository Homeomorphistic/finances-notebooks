# inflation.py
"""A module containing base for inflation.

It contains closed-form formulas for basic ideas in finances and also their empirical computations used for testing.

Not to be used by a user.
"""
from typing import Union

from numpy import ndarray

from compound_interest import _compound


def _inflate(d_n: Union[float, ndarray[float]],
             n: Union[float, ndarray[float]],
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