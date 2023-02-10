import pytest
from numpy.testing import assert_allclose
from numpy import array

from finances_scripts.compound_interest import _compound, _compound_recursive_helper, _compound_iterative_helper, \
    _compound_recursive, _compound_iterative, _compound_frequently


@pytest.mark.parametrize("d_0, n, p, d_n", [
    (1000, 1, 0.05, 1050),
    (array([-1000, 0, 1000]), 1, 0.05, array([-1050, 0, 1050])),
    (1000, array([-1, 0, 1]), 0.05, array([952.381, 1000, 1050])),
    (1000, 1, array([-0.05, 0, 0.05]), array([950, 1000, 1050])),
    (array([-1000, 0, 1000]), array([-1, 0, 1]), array([-0.05, 0, 0.05]), array([-1052.6315, 0, 1050]))
])
def test_compound_closed_formula(d_0, n, p, d_n):
    assert_allclose(actual=_compound(d_0, n, p), desired=d_n,
                    err_msg="Compound closed formula value does not match desired." )


@pytest.mark.parametrize("d_0, n, p", [
    (1, array([1, 2]), array([1, 2, 3])),
    (array([1, 2]), array([1, 2]), array([1, 2, 3]))
])
def test_compound_closed_formula_raises_error_when_nonbroadcastable(d_0, n, p):
    with pytest.raises(ValueError):
        _compound(d_0, n, p)


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
def test_compound_recursive_helper(d_0, n, p):
    assert_allclose(actual=_compound_recursive_helper(d_0, n, p), desired=_compound(d_0, n, p),
                    err_msg="Compound recursive algorithm in helper does not match closed formula.")


@pytest.mark.parametrize("d_0, n, p", [
    (1, array([1, 2]), array([1, 2, 3])),
    (array([1, 2]), array([1, 2]), array([1, 2, 3]))
])
def test_compound_recursive_helper_raises_error_when_nonbroadcastable(d_0, n, p):
    with pytest.raises(ValueError):
        _compound_recursive_helper(d_0, n, p)


@pytest.mark.parametrize("d_0, n, p", [
    (1000, array([-2, 0, 4]), 0.05),
    (array([-1000, 0, 1000]), array([-3, 0, 3]), array([-0.05, 0, 0.05])),
])
def test_compound_recursive(d_0, n, p):
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
def test_compound_iterative_helper(d_0, n, p):
    assert_allclose(actual=_compound_iterative_helper(d_0, n, p), desired=_compound(d_0, n, p),
                    err_msg="Compound iterative algorithm in helper does not match closed formula.")


@pytest.mark.parametrize("d_0, n, p", [
    (1, array([1, 2]), array([1, 2, 3])),
    (array([1, 2]), array([1, 2]), array([1, 2, 3]))
])
def test_compound_iterative_helper_raises_error_when_nonbroadcastable(d_0, n, p):
    with pytest.raises(ValueError):
        _compound_iterative_helper(d_0, n, p)


@pytest.mark.parametrize("d_0, n, p", [
    (1000, array([-2, 0, 4]), 0.05),
    (array([-1000, 0, 1000]), array([-3, 0, 3]), array([-0.05, 0, 0.05])),
])
def test_compound_iterative(d_0, n, p):
    assert_allclose(actual=_compound_iterative(d_0, n, p), desired=_compound(d_0, n, p),
                    err_msg="Compound recursive algorithm does not match closed formula.")


@pytest.mark.parametrize("d_0, n, p, m", [
    (1000, 2, 0.05, 2),
    (array([-1000, 0, 1000]), 2, 0.05, 2),
    (1000, array([-2, 0, 4]), 0.05, 2),
    (1000, 5, array([-0.05, 0, 0.05]), 2),
    (1000, 5, 0.05, array([1, 2, 3])),
    (array([-1000, 0, 1000]), [-3, 0, 3], array([-0.05, 0, 0.05]), array([1, 2, 3]))
])
def test_compound_frequently(d_0, n, p, m):
    assert_allclose(actual=_compound_frequently(d_0, n, p, m), desired=_compound_iterative(d_0, n*m, p/m),
                    err_msg="Compounding frequently does not match iterative method.")
    assert_allclose(actual=_compound_frequently(d_0, n, p, m), desired=_compound_recursive(d_0, n*m, p/m),
                    err_msg="Compounding frequently does not match recursive method.")


@pytest.mark.parametrize("d_0, n, p, m", [
    (1000, 2, 0.05, -2),
    (array([-1000, 0, 1000]), [-3, 0, 3], array([-0.05, 0, 0.05]), array([-1, 2, 3])),
    (1000, 2, 0.05, 0),
    (array([-1000, 0, 1000]), [-3, 0, 3], array([-0.05, 0, 0.05]), array([1, 0, 3]))
])
def test_compound_frequently_raises_error_when_negative_or_zero_frequency(d_0, n, p, m):
    with pytest.raises(AssertionError):
        _compound_frequently(d_0, n, p, m)