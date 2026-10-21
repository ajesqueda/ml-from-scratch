import math

import numpy as np
import pytest
from mle import (
    classify_step_size,
    coin_log_likelihood,
    coin_mle_closed_form,
    coin_mle_grid,
    fit_regression_by_mle,
    gaussian_log_likelihood,
    gaussian_mle,
    gaussian_regression_nll,
    gradient_descent_1d,
    least_squares_normal_equations,
    product_vs_log_sum,
)


@pytest.mark.parametrize("heads,n", [(7, 10), (1, 4), (0, 5), (5, 5), (23, 100)])
def test_grid_search_agrees_with_closed_form(heads, n):
    assert coin_mle_grid(heads, n) == pytest.approx(coin_mle_closed_form(heads, n), abs=1e-3)


def test_p_point_seven_explains_seven_heads_better_than_fair_coin():
    assert coin_log_likelihood(0.7, 7, 10) > coin_log_likelihood(0.5, 7, 10)


def test_edge_cases_do_not_produce_nan():
    assert coin_log_likelihood(0.0, 0, 5) == 0.0  # all tails, p = 0 is a perfect fit
    assert coin_log_likelihood(0.0, 2, 5) == -math.inf  # impossible, not nan
    with pytest.raises(ValueError):
        coin_log_likelihood(1.5, 1, 2)


def test_product_underflows_but_log_sum_does_not():
    product, log_sum = product_vs_log_sum([0.01] * 200)
    assert product == 0.0
    assert log_sum == pytest.approx(200 * math.log(0.01))


def test_gaussian_mle_is_mean_and_biased_variance():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    mu, sigma2 = gaussian_mle(x)
    assert mu == 5.0
    assert sigma2 == pytest.approx(np.var(x))  # ddof=0, divides by n
    # Nudging either parameter away from the MLE lowers the likelihood.
    best = gaussian_log_likelihood(x, mu, sigma2)
    assert best > gaussian_log_likelihood(x, mu + 0.1, sigma2)
    assert best > gaussian_log_likelihood(x, mu, sigma2 * 1.1)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = np.column_stack([np.ones(300), rng.normal(size=300), rng.normal(size=300)])
    y = X @ np.array([0.5, 2.0, -1.0]) + rng.normal(0, 0.3, 300)
    return X, y


def test_normal_equations_match_numpy_lstsq(regression_data):
    X, y = regression_data
    expected, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(least_squares_normal_equations(X, y), expected, atol=1e-10)


def test_mle_by_gradient_descent_recovers_least_squares(regression_data):
    X, y = regression_data
    np.testing.assert_allclose(
        fit_regression_by_mle(X, y), least_squares_normal_equations(X, y), atol=1e-6
    )


@pytest.mark.parametrize("sigma2", [0.1, 1.0, 10.0])
def test_nll_minimizer_does_not_depend_on_sigma(regression_data, sigma2):
    X, y = regression_data
    w_star = least_squares_normal_equations(X, y)
    for direction in np.eye(3):
        assert gaussian_regression_nll(w_star, X, y, sigma2) < gaussian_regression_nll(
            w_star + 0.01 * direction, X, y, sigma2
        )


@pytest.mark.parametrize(
    "lr,expected",
    [(0.1, "converges"), (0.5, "one-step"), (1.0, "oscillates"), (1.1, "diverges")],
)
def test_step_size_behaviour_on_a_bowl(lr, expected):
    path = gradient_descent_1d(lambda x: 2 * (x - 3), 0.0, lr, steps=100)
    assert classify_step_size(path, target=3.0) == expected
