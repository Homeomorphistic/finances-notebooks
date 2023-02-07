import pytest
from numpy.testing import assert_allclose
from numpy import array

from finances_scripts.compound_interest import _compound, _compound_recursive, _compound_iterative


@pytest.mark.parametrize("d_0, n, p, d_n", [
    (1000, 1, 0.05, 1050),
    (array([-1000, 0, 1000]), 1, 0.05, array([-1050, 0, 1050])),
    (1000, array([-1, 0, 1]), 0.05, array([952.381, 1000, 1050])),
    (1000, 1, array([-0.05, 0, 0.05]), array([950, 1000, 1050])),
    (array([-1000, 0, 1000]), array([-1, 0, 1]), array([-0.05, 0, 0.05]), array([-1052.6315, 0, 1050]))
])
def test_compounding_closed_formula(d_0, n, p, d_n):
    assert_allclose(actual=_compound(d_0, n, p), desired=d_n,
                    err_msg="Compound closed formula value does not match desired." )


@pytest.mark.parametrize("d_0, n, p", [
    (1000, 2, 0.05),
    (array([-1000, 0, 1000]), 2, 0.05),
    (1000, -2, 0.05),
    (1000, 0, 0.05),
    (1000, 4, 0.05),
    (1000, 5, array([-0.05, 0, 0.05])),
    (array([-1000, 0, 1000]), -3, array([-0.05, 0, 0.05])),
    (array([-1000, 0, 1000]), 0, array([-0.05, 0, 0.05])),
    (array([-1000, 0, 1000]), 3, array([-0.05, 0, 0.05]))
])
def test_compounding_recursive(d_0, n, p):
    assert_allclose(actual=_compound_recursive(d_0, n, p), desired=_compound(d_0, n, p),
                    err_msg="Compound recursive algorithm does not match closed formula.")


@pytest.mark.parametrize("d_0, n, p", [
    (1000, 2, 0.05),
    (array([-1000, 0, 1000]), 2, 0.05),
    (1000, -2, 0.05),
    (1000, 0, 0.05),
    (1000, 4, 0.05),
    (1000, 5, array([-0.05, 0, 0.05])),
    (array([-1000, 0, 1000]), -3, array([-0.05, 0, 0.05])),
    (array([-1000, 0, 1000]), 0, array([-0.05, 0, 0.05])),
    (array([-1000, 0, 1000]), 3, array([-0.05, 0, 0.05]))
])
def test_compounding_iterative(d_0, n, p):
    assert_allclose(actual=_compound_iterative(d_0, n, p), desired=_compound(d_0, n, p),
                    err_msg="Compound iterative algorithm does not match closed formula.")

