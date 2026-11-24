from fractions import Fraction

import numpy as np
import pytest
from chains import (
    check_stochastic,
    distribution_after,
    mixing_factor,
    normalize,
    power_via_eigen,
    steady_state,
    swap_and_flip,
)

rng = np.random.default_rng(1118)


def random_chain(n):
    """Random column-stochastic matrix with exact fractions and no zero entries."""
    counts = rng.integers(1, 10, size=(n, n))
    return [[Fraction(int(counts[i, j]), int(counts[:, j].sum())) for j in range(n)]
            for i in range(n)]


def matvec(P, s):
    return [sum(P[i][j] * s[j] for j in range(len(s))) for i in range(len(P))]


# --- steady state ----------------------------------------------------------------------

def test_two_state_example_is_exact():
    P = [[Fraction(9, 10), Fraction(1, 2)], [Fraction(1, 10), Fraction(1, 2)]]
    assert steady_state(P) == [Fraction(5, 6), Fraction(1, 6)]


def test_decimal_floats_are_read_exactly():
    assert steady_state([[0.8, 0.3], [0.2, 0.7]]) == [Fraction(3, 5), Fraction(2, 5)]


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("trial", range(10))
def test_steady_state_is_fixed_and_sums_to_one(n, trial):
    P = random_chain(n)
    s = steady_state(P)
    assert matvec(P, s) == s
    assert sum(s) == 1
    assert all(x > 0 for x in s)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_matches_numpy_eigenvector_for_lambda_1(n):
    P = random_chain(n)
    vals, vecs = np.linalg.eig(np.array(P, dtype=float))
    v = vecs[:, int(np.argmin(np.abs(vals - 1)))].real
    assert [float(x) for x in steady_state(P)] == pytest.approx(list(v / v.sum()))


def test_long_run_matches_steady_state_from_any_start():
    P = random_chain(3)
    s = steady_state(P)
    for start in ([1, 0, 0], [0, 0, 1], [Fraction(1, 3)] * 3):
        after = distribution_after(P, start, 60)
        assert [float(x) for x in after] == pytest.approx([float(x) for x in s], abs=1e-9)


def test_swap_and_flip_solves_the_one_equation():
    for a, b in rng.integers(-9, 10, size=(20, 2)).tolist():
        if a == b == 0:
            continue
        s1, s2 = swap_and_flip(a, b)
        assert a * s1 + b * s2 == 0
        assert (s1, s2) != (0, 0)


def test_normalize_divides_by_the_whole_sum_and_keeps_order():
    assert normalize([3, 2]) == [Fraction(3, 5), Fraction(2, 5)]
    assert normalize([Fraction(3, 10), Fraction(2, 10)]) == [Fraction(3, 5), Fraction(2, 5)]
    with pytest.raises(ValueError, match="sum to 0"):
        normalize([1, -1])


# --- chains that don't have one answer -------------------------------------------------

def test_identity_has_no_single_steady_state():
    with pytest.raises(ValueError, match="identity"):
        steady_state([[1, 0], [0, 1]])


def test_split_chain_has_two_steady_states():
    P = [[1, 0, 0], [0, Fraction(1, 2), Fraction(1, 2)], [0, Fraction(1, 2), Fraction(1, 2)]]
    with pytest.raises(ValueError, match="2 independent steady states"):
        steady_state(P)


def test_flip_flop_chain_has_a_steady_state_but_never_settles():
    P = [[0, 1], [1, 0]]
    assert steady_state(P) == [Fraction(1, 2), Fraction(1, 2)]
    assert mixing_factor(P) == pytest.approx(1)  # λ2 = −1: the start is never forgotten
    assert distribution_after(P, [1, 0], 101) == [0, 1]


def test_rows_instead_of_columns_is_caught():
    with pytest.raises(ValueError, match="columns are 'from'"):
        check_stochastic([[Fraction(9, 10), Fraction(1, 10)], [Fraction(1, 2), Fraction(1, 2)]])


def test_negative_probability_is_caught():
    with pytest.raises(ValueError, match="negative"):
        check_stochastic([[Fraction(3, 2), 0], [Fraction(-1, 2), 1]])


# --- mixing --------------------------------------------------------------------------------

def test_mixing_factor_is_how_fast_the_start_fades():
    P = [[Fraction(9, 10), Fraction(1, 2)], [Fraction(1, 10), Fraction(1, 2)]]
    assert mixing_factor(P) == pytest.approx(0.4)
    s = steady_state(P)
    gaps = [abs(float(distribution_after(P, [0, 1], k)[0] - s[0])) for k in range(1, 8)]
    for g1, g2 in zip(gaps, gaps[1:], strict=False):
        assert g2 / g1 == pytest.approx(0.4)


# --- A^k y via eigen-pieces ----------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 1, 2, 5, 9])
def test_power_via_eigen_matches_multiplying(k):
    A = [[4, 1], [2, 3]]
    y = [7, 1]
    expected = np.linalg.matrix_power(np.array(A), k) @ np.array(y)
    assert power_via_eigen(A, y, k) == expected.tolist()


def test_power_via_eigen_with_negative_eigenvalue():
    A = [[-4, 3], [2, 1]]  # λ = −5, 2
    for k in range(6):
        expected = np.linalg.matrix_power(np.array(A), k) @ np.array([1, 4])
        assert power_via_eigen(A, [1, 4], k) == expected.tolist()


def test_power_via_eigen_needs_distinct_rational_eigenvalues():
    with pytest.raises(ValueError, match="distinct"):
        power_via_eigen([[3, 1], [0, 3]], [1, 1], 2)
    with pytest.raises(ValueError, match="rational"):
        power_via_eigen([[2, 1], [1, 1]], [1, 1], 2)


# --- eigenvalue rules ---------------------------------------------------------------------------

@pytest.mark.parametrize("trial", range(10))
def test_eigenvalue_rules_and_the_trace_det_check(trial):
    A = rng.integers(-5, 6, size=(3, 3)).astype(float)
    while abs(np.linalg.det(A)) < 0.5:
        A = rng.integers(-5, 6, size=(3, 3)).astype(float)
    lam = np.linalg.eigvals(A)

    def same(M, expected):
        got = np.sort_complex(np.round(np.linalg.eigvals(M), 6))
        return np.allclose(got, np.sort_complex(np.round(expected, 6)), atol=1e-5)

    assert same(A @ A, lam**2)
    assert same(A + 3 * np.eye(3), lam + 3)
    assert same(-2 * A, -2 * lam)
    assert same(np.linalg.inv(A), 1 / lam)
    # check A² without solving its polynomial: trace = Σλ², det = Πλ²
    assert np.trace(A @ A) == pytest.approx(np.sum(lam**2).real)
    assert np.linalg.det(A @ A) == pytest.approx(np.prod(lam**2).real)


# --- bad or degenerate input -----------------------------------------------------------

def test_empty_matrix_is_rejected():
    with pytest.raises(ValueError, match="at least one state"):
        check_stochastic([])


def test_swap_and_flip_has_nothing_to_solve_for_two_zeros():
    with pytest.raises(ValueError, match="no equation"):
        swap_and_flip(0, 0)


def test_floats_are_read_as_the_decimals_they_print_as():
    assert normalize([0.3, 0.2]) == [Fraction(3, 5), Fraction(2, 5)]
    P = [[0.9, 0.5], [0.1, 0.5]]
    assert distribution_after(P, [0.5, 0.5], 1) == [Fraction(7, 10), Fraction(3, 10)]


def test_distribution_after_rejects_negative_steps_and_wrong_lengths():
    P = [[Fraction(9, 10), Fraction(1, 2)], [Fraction(1, 10), Fraction(1, 2)]]
    with pytest.raises(ValueError, match="negative"):
        distribution_after(P, [1, 0], -1)
    with pytest.raises(ValueError, match="entries but the chain has 2"):
        distribution_after(P, [1, 0, 0], 3)
    with pytest.raises(ValueError, match="entries but the chain has 2"):
        distribution_after(P, [1], 3)


def test_one_state_chain_has_nothing_to_forget():
    assert mixing_factor([[1]]) == 0.0
    assert steady_state([[1]]) == [1]


def test_power_via_eigen_keeps_float_matrices_exact():
    assert power_via_eigen([[4.0, 1.0], [2.0, 3.0]], [1, 0], 3) == \
        power_via_eigen([[4, 1], [2, 3]], [1, 0], 3)


@pytest.mark.parametrize("y, k", [([1, 0, 0], 2), ([1, 0], -1)])
def test_power_via_eigen_rejects_bad_input(y, k):
    with pytest.raises(ValueError):
        power_via_eigen([[4, 1], [2, 3]], y, k)
