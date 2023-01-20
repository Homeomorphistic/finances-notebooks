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
    # TODO contrib_periods =0 => contrib=0
    if isinstance(contribution_periods, ndarray):  # multiple contribution periods
        zero_contrib = (contribution == 0)
        non_zero_contrib = (contribution != 0)

        d_n = initial_deposit

        # Check if any parameter is a vector and use only nonzero contribution
        initial_deposit_nz = (initial_deposit[non_zero_contrib] if isinstance(initial_deposit, ndarray)
                              else initial_deposit)
        periods_nz = periods[non_zero_contrib] if isinstance(periods, ndarray) else periods
        interest_nz = interest[non_zero_contrib] if isinstance(interest, ndarray) else interest
        frequency_nz = frequency[non_zero_contrib] if isinstance(frequency, ndarray) else frequency
        contribution_nz = contribution[non_zero_contrib] if isinstance(contribution, ndarray) else contribution
        contribution_periods_nz = (contribution_periods[non_zero_contrib] if isinstance(contribution_periods, ndarray)
                                   else contribution_periods)
        tax_nz = tax[non_zero_contrib] if isinstance(tax, ndarray) else tax

        # The same for zero contribution
        initial_deposit_z = (initial_deposit[zero_contrib] if isinstance(initial_deposit, ndarray)
                             else initial_deposit)
        periods_z = periods[zero_contrib] if isinstance(periods, ndarray) else periods
        interest_z = interest[zero_contrib] if isinstance(interest, ndarray) else interest
        frequency_z = frequency[zero_contrib] if isinstance(frequency, ndarray) else frequency
        tax_z = tax[zero_contrib] if isinstance(tax, ndarray) else tax

        # First compute deposit after periods of contribution
        d_n[non_zero_contrib] = _compound_and_contribute_frequently(d_0=initial_deposit_nz,
                                                                    n=contribution_periods_nz + 1,
                                                                    p=interest_nz * (1 - tax_nz),
                                                                    m=frequency_nz, c=contribution_nz)
        # Then compute the rest
        d_n[non_zero_contrib] = _compound_frequently(d_0=d_n[non_zero_contrib],
                                                     n=periods_nz - contribution_periods_nz - 1,
                                                     p=interest_nz * (1 - tax_nz), m=frequency_nz)
        d_n[zero_contrib] = _compound_frequently(d_0=initial_deposit_z, n=periods_z, p=interest_z * (1 - tax_z),
                                                 m=frequency_z)

    else:  # one contribution period
        if contribution_periods == 0: # no contributions made, exponential formula
            d_n = _compound_frequently(d_0=initial_deposit, n=periods, p=interest*(1-tax), m=frequency)
        else:  # with contribution, geometric series formula
            # First compute deposit after periods of contribution
            d_c_n = _compound_and_contribute_frequently(d_0=initial_deposit, n=contribution_periods+1, p=interest*(1-tax),
                                                        m=frequency, c=contribution)
            # Then compute the rest
            d_n = _compound_frequently(d_0=d_c_n, n=periods-contribution_periods-1, p=interest*(1-tax), m=frequency)

    # Then inflate if needed
    return d_n
    # return _inflate_frequently(d_n=d_n, n=periods, f=inflation, m=frequency)


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
