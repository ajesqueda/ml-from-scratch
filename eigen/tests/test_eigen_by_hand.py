from fractions import Fraction

import numpy as np
import pytest
from eigen_by_hand import (
    NoRealEigenvalues,
    eig2x2,
    eigenpairs,
    eigenvector,
    power_iteration,
    trace_det,
)

rng = np.random.default_rng(1217)


def matvec(A, v):
    return [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]]


def is_eigenpair(A, lam, v, tol=1e-9):
    Av = np.array(matvec(A, v), dtype=float)
    return np.allclose(Av, float(lam) * np.array(v, dtype=float), atol=tol)


def parallel(u, v, tol=1e-6):
    return abs(float(u[0]) * float(v[1]) - float(u[1]) * float(v[0])) < tol * (
        np.linalg.norm(np.array(u, float)) * np.linalg.norm(np.array(v, float)))


def random_real_2x2():
    while True:
        A = rng.integers(-6, 7, size=(2, 2)).tolist()
        t, p = trace_det(A)
        if (t / 2) ** 2 - p > 0:  # two distinct real eigenvalues
            return A


# --- eigenvalues ------------------------------------------------------------------------

def test_clean_example_is_exact():
    assert eig2x2([[4, 1], [2, 3]]) == [5, 2]
    assert all(isinstance(lam, Fraction) for lam in eig2x2([[4, 1], [2, 3]]))


def test_sign_flip_drill():
    # λ² + 3λ − 10 = (λ + 5)(λ − 2): the factor (λ + 5) means λ = −5
    assert eig2x2([[-4, 3], [2, 1]]) == [-5, 2]  # biggest |λ| first, sign kept


def test_irrational_eigenvalues_fall_back_to_floats():
    lams = eig2x2([[2, 1], [1, 1]])
    assert lams == pytest.approx([(3 + 5**0.5) / 2, (3 - 5**0.5) / 2])


@pytest.mark.parametrize("trial", range(40))
def test_sum_is_trace_product_is_det(trial):
    A = random_real_2x2()
    lam1, lam2 = eig2x2(A)
    t, p = trace_det(A)
    assert float(lam1 + lam2) == pytest.approx(float(t))
    assert float(lam1 * lam2) == pytest.approx(float(p))


@pytest.mark.parametrize("trial", range(40))
def test_matches_numpy(trial):
    A = random_real_2x2()
    assert sorted(float(x) for x in eig2x2(A)) == pytest.approx(sorted(np.linalg.eigvals(A).real))


def test_triangular_eigenvalues_are_the_diagonal():
    assert sorted(eig2x2([[7, 4], [0, -2]])) == [-2, 7]
    assert sorted(eig2x2([[-3, 0], [9, 1]])) == [-3, 1]


def test_fractional_eigenvalues_stay_exact():
    A = [[Fraction(1, 2), Fraction(1, 3)], [0, Fraction(1, 4)]]
    assert eig2x2(A) == [Fraction(1, 2), Fraction(1, 4)]


def test_rotation_has_no_real_eigenvalues():
    with pytest.raises(NoRealEigenvalues):
        eig2x2([[0, -1], [1, 0]])
    theta = 0.7
    with pytest.raises(NoRealEigenvalues):
        eig2x2([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


# --- eigenvectors -------------------------------------------------------------------------

def test_clean_eigenvectors_are_whole_numbers():
    pairs = dict(eigenpairs([[4, 1], [2, 3]]))
    assert pairs[5] == [[1, 1]]
    assert pairs[2] == [[1, -2]]  # same line as the note's (−0.5, 1)


@pytest.mark.parametrize("trial", range(40))
def test_every_pair_satisfies_Av_eq_lambda_v(trial):
    A = random_real_2x2()
    for lam, vecs in eigenpairs(A):
        for v in vecs:
            assert any(x != 0 for x in v)  # never the zero vector
            assert is_eigenpair(A, lam, v)


@pytest.mark.parametrize("trial", range(40))
def test_eigenvectors_match_numpy_up_to_scale(trial):
    A = random_real_2x2()
    vals, vecs = np.linalg.eig(np.array(A, dtype=float))
    for lam, (v,) in eigenpairs(A):
        col = vecs[:, int(np.argmin(np.abs(vals - float(lam))))].real  # real eigenvalues
        assert parallel(v, col)


def test_wrong_lambda_is_rejected():
    with pytest.raises(ValueError, match="isn't an eigenvalue"):
        eigenvector([[4, 1], [2, 3]], 3)


def test_first_row_all_zero_uses_the_second_row():
    # A − 2I = [[0, 0], [1, 0]]: the first row says nothing, the second says x1 = 0
    assert eigenvector([[2, 0], [1, 2]], 2) == [[0, 1]]


def test_shear_has_one_eigenvector_for_a_repeated_eigenvalue():
    pairs = eigenpairs([[3, 1], [0, 3]])
    assert pairs == [(3, [[1, 0]])]


def test_scalar_matrix_makes_every_vector_an_eigenvector():
    assert eigenpairs([[3, 0], [0, 3]]) == [(3, [[1, 0], [0, 1]])]
    for v in rng.integers(-5, 6, size=(5, 2)).tolist():
        assert is_eigenpair([[3, 0], [0, 3]], 3, v)


def test_irrational_eigenvectors_are_unit_length():
    for lam, (v,) in eigenpairs([[2, 1], [1, 1]]):
        assert np.linalg.norm(v) == pytest.approx(1)
        assert is_eigenpair([[2, 1], [1, 1]], lam, v)


# --- power iteration ---------------------------------------------------------------------

@pytest.mark.parametrize("A, lam1", [([[4, 1], [2, 3]], 5), ([[-4, 3], [2, 1]], -5)])
def test_power_iteration_finds_lambda_1_with_its_sign(A, lam1):
    lam, v = power_iteration(A)
    assert lam == pytest.approx(lam1)
    assert is_eigenpair(A, lam1, v, tol=1e-8)


def test_power_iteration_converges_at_rate_lambda2_over_lambda1():
    A = [[4, 1], [2, 3]]  # |λ2 / λ1| = 2/5
    errors = [abs(power_iteration(A, steps=k, check=False)[0] - 5) for k in (5, 10, 15)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[0] < (2 / 5) ** 4  # five more steps shrink it by > (2/5)^4


# --- floats and power-iteration failure modes -------------------------------------------

def test_close_float_eigenvalues_stay_apart():
    lams = eig2x2([[1e8, 1.0], [1.0, 1e8]])
    assert lams == pytest.approx([1e8 + 1, 1e8 - 1], abs=1e-6)


def test_small_eigenvalue_survives_a_huge_one():
    assert eig2x2([[1e20, 0.0], [0.0, 1.0]]) == pytest.approx([1e20, 1.0])


def test_tiny_float_matrices_get_real_eigenvectors():
    A = [[2e-12, 1e-12], [1e-12, 2e-12]]  # eigenvalues 3e-12 and 1e-12
    with pytest.raises(ValueError, match="isn't an eigenvalue"):
        eigenvector(A, 5e-12)
    (v,) = eigenvector(A, 3e-12)
    Av = matvec(A, v)
    assert np.allclose(Av, [3e-12 * x for x in v], atol=1e-24)


@pytest.mark.parametrize("steps", [0, -3])
def test_power_iteration_needs_at_least_one_step(steps):
    with pytest.raises(ValueError, match="at least 1"):
        power_iteration([[2, 0], [0, 1]], steps=steps)


def test_power_iteration_reports_hitting_zero():
    with pytest.raises(ValueError, match="sent the vector to 0"):
        power_iteration([[0, 0], [0, 0]])


@pytest.mark.parametrize("A", [[[1, 0], [0, -1]],     # |λ| tie
                               [[0, -1], [1, 0]]])    # rotation: complex pair
def test_power_iteration_refuses_when_there_is_no_single_biggest(A):
    with pytest.raises(ValueError, match="didn't settle"):
        power_iteration(A)
