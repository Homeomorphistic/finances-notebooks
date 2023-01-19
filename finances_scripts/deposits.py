# deposits.py
"""A module containing calculations regarding different types of deposit."""
from typing import Union

from numpy import ndarray

from .compound_interest import _compound_frequently, _compound_iterate
from .inflation import _inflate_frequently, _inflate_iterate
from .contributions import _compound_and_contribute_frequently, _compound_and_contribute_iterate


def deposit(initial_deposit: Union[float, ndarray[float]],
            periods: Union[int, ndarray[int]],
            interest: Union[float, ndarray[float]],
            frequency: Union[int, ndarray[int]] = 1,
            contribution: Union[float, ndarray[float]] = 0,
            contribution_periods: Union[int, ndarray[int]] = 0,
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
    :param contribution_periods: number of periodic contribution or matrix of numbers of contributions
    :param inflation: periodic inflation rate or matrix of inflations
    :param tax: periodic tax or matrix of taxes
    :return: deposit after periods or matrix of deposits
    """
    # First compute deposit after periods of contribution
    d_c_n = _compound_and_contribute_frequently(d_0=initial_deposit, n=contribution_periods+1, p=interest*(1-tax),
                                                m=frequency, c=contribution)
    # Then compute the rest
    d_n = _compound_frequently(d_0=d_c_n, n=periods-contribution_periods-1, p=interest*(1-tax), m=frequency)
    # Then inflate if needed
    return _inflate_frequently(d_n=d_n, n=periods, f=inflation, m=frequency)


def _deposit_iterate(initial_deposit: Union[float, ndarray[float]],
                     periods: Union[int, ndarray[int]],
                     interest: Union[float, ndarray[float]],
                     frequency: Union[int, ndarray[int]] = 1,
                     contribution: Union[float, ndarray[float]] = 0,
                     contribution_periods: Union[int, ndarray[int]] = 0,
                     inflation: Union[float, ndarray[float]] = 0,
                     tax: Union[float, ndarray[float]] = .19,
                     ) -> Union[float, ndarray[float]]:
    """Calculate compound interest of initial deposit after periods with contributions, accounting for tax
    and inflation, using iterative method.

    This method exists solely for testing purpose. Matrices have to be broadcastable to obtain any result.

    :param initial_deposit: initial deposit or matrix of initial deposits
    :param periods: number of periods or matrix of numbers
    :param interest: periodic interest rate or matrix of interests
    :param frequency: compounding frequency or matrix of frequencies
    :param contribution: periodic contribution or matrix of contributions
    :param contribution_periods: number of periodic contribution or matrix of numbers of contributions
    :param inflation: periodic inflation rate or matrix of inflations
    :param tax: periodic tax or matrix of taxes
    :return: deposit after periods or matrix of deposits
    """
    # First compute deposit after periods of contribution
    d_c_n = _compound_and_contribute_iterate(d_0=initial_deposit, n=contribution_periods*frequency+1,
                                             p=interest * (1 - tax) / frequency, c=contribution/frequency)
    # Then compute the rest
    d_n = _compound_iterate(d_0=d_c_n, n=(periods-contribution_periods)*frequency-1,
                            p=interest * (1 - tax) / frequency)
    # Then inflate if needed
    return _inflate_iterate(d_n=d_n, n=periods*frequency, f=inflation/frequency)
