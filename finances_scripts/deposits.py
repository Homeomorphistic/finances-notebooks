# deposits.py
"""A module containing calculations regarding different types of deposit."""
from typing import Union

from numpy import ndarray

from .compound_interest import _compound_frequently
from .inflation import _inflate_frequently
from .contributions import _compound_and_contribute_frequently


def deposit(initial_deposit: Union[float, ndarray[float]],
            periods: Union[int, ndarray[int]],
            interest: Union[float, ndarray[float]],
            frequency: Union[int, ndarray[int]] = 1,
            contribution: Union[float, ndarray[float]] = 0,
            inflation: Union[float, ndarray[float]] = 0,
            tax: Union[float, ndarray[float]] = .19,
            ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit after periods with contributions, accounting for tax and inflation.

    Matrices have to be broadcastable to obtain any result.

    :param initial_deposit: initial deposit or matrix of initial deposits
    :param periods: number of periods or matrix of numbers
    :param interest: periodic interest rate or matrix of interests
    :param frequency: compounding frequency or matrix of frequencies
    :param contribution: periodic contribution or matrix of contributions
    :param inflation: periodic inflation rate or matrix of inflations
    :param tax: periodic tax or matrix of taxes
    :return: deposit after periods or matrix of deposits
    """
    d_n = _compound_and_contribute_frequently(d_0=initial_deposit, n=periods, p=interest*(1-tax), m=frequency,
                                              c=contribution)
    return _inflate_frequently(d_n=d_n, n=periods, f=inflation, m=frequency)
