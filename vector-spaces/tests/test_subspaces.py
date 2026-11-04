from fractions import Fraction

import numpy as np
import pytest
from gauss import rank
from subspaces import (
    as_integers,
    component_variances,
    contains,
    dim_sum,
    equations_for_span,
    intersect,
    null_space,
    pivot_report,
)

rng = np.random.default_rng(603)

MATRICES = [
    [[1, 2, 0, 3, 1], [2, 4, 1, 7, 1], [1, 2, 1, 4, 0]],
    [[2, -1, 3]],
    [[1, 0], [0, 1]],
    [[1, 1, 2, 0], [0, 1, 1, 3], [1, 2, 3, 3], [2, 3, 5, 3]],
    [[0, 0, 0], [0, 0, 0]],
    [[3, 6, -9, 0, 2], [1, 2, -3, 1, 1]],
]


def matvec(A, x):
    return [sum(Fraction(a) * v for a, v in zip(row, x, strict=True)) for row in A]


# --- null space ----------------------------------------------------------------

@pytest.mark.parametrize("A", MATRICES)
def test_every_basis_vector_solves_Ax_eq_0(A):
    for v in null_space(A):
        assert matvec(A, v) == [0] * len(A)


@pytest.mark.parametrize("A", MATRICES)
def test_dimension_is_columns_minus_rank(A):
    basis = null_space(A)
    assert len(basis) == len(A[0]) - rank(A)
    if basis:
        assert rank(basis) == len(basis)  # the vectors are independent


@pytest.mark.parametrize("seed", range(20))
def test_random_low_rank_matrices(seed):
    local = np.random.default_rng(seed)
    m, n = local.integers(2, 6, size=2)
    r = int(local.integers(1, min(m, n) + 1))
    A = (local.integers(-4, 5, size=(m, r)) @ local.integers(-4, 5, size=(r, n))).tolist()
    basis = null_space(A)
    assert len(basis) == n - rank(A)
    assert all(matvec(A, v) == [0] * m for v in basis)


def test_full_column_rank_has_only_zero():
    assert null_space([[1, 2], [3, 4], [5, 6]]) == []


def test_basis_is_whole_numbers_in_lowest_terms():
    for v in null_space([[2, 3, 5], [4, 1, 7]]):
        assert all(isinstance(x, int) for x in v)
        assert np.gcd.reduce(v) == 1


def test_as_integers_clears_fractions_and_common_factors():
    assert as_integers([Fraction(1, 2), Fraction(-1, 3), 0]) == [3, -2, 0]
    assert as_integers([4, 6, -8]) == [2, 3, -4]
    assert as_integers([0, 0]) == [0, 0]


# --- span -> equations -----------------------------------------------------------

@pytest.mark.parametrize("G", [
    [[3, 1, -2], [1, 4, 5]],
    [[1, 0, 2, 0], [0, 1, 1, 1]],
    [[1, 2, 3, 4, 5]],
    [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
])
def test_equations_vanish_on_every_generator(G):
    eqs = equations_for_span(G)
    n, m = len(G[0]), rank(G)
    assert len(eqs) == n - m
    for g in G:
        assert contains(eqs, g)
    for a, b in rng.integers(-5, 6, size=(5, 2)):  # and on combinations of them
        combo = [int(a) * x + int(b) * y for x, y in zip(G[0], G[-1], strict=True)]
        assert contains(eqs, combo)


def test_three_d_plane_matches_cross_product():
    for _ in range(10):
        g1, g2 = rng.integers(-6, 7, size=(2, 3))
        if not np.cross(g1, g2).any():
            continue
        (a,) = equations_for_span([g1.tolist(), g2.tolist()])
        assert not np.cross(a, np.cross(g1, g2)).any()  # parallel: same plane


def test_span_to_equations_and_back_recovers_the_span():
    G = [[1, 0, 2, 0, 1], [0, 1, 1, 1, 0]]
    back = null_space(equations_for_span(G))
    assert rank(back) == rank(G) == rank([*G, *back])


def test_a_vector_outside_the_span_fails_the_equations():
    eqs = equations_for_span([[1, 0, 2], [0, 1, 1]])
    assert contains(eqs, [2, 3, 7])
    assert not contains(eqs, [0, 0, 1])


def test_whole_space_needs_no_equations():
    assert equations_for_span([[1, 0], [0, 1]]) == []
    assert contains([], [5, -2])


# --- intersections -----------------------------------------------------------------

def test_two_planes_in_r3_meet_in_a_line():
    U = [[0, 0, 1]]   # z = 0
    W = [[1, -1, 0]]  # x = y
    (line,) = intersect(U, W)
    assert contains(U, line) and contains(W, line)
    assert as_integers(line) in ([1, 1, 0], [-1, -1, 0])


@pytest.mark.parametrize("seed", range(15))
def test_dimension_formula(seed):
    local = np.random.default_rng(100 + seed)
    n = 5
    gU = local.integers(-3, 4, size=(int(local.integers(1, 4)), n)).tolist()
    gW = local.integers(-3, 4, size=(int(local.integers(1, 4)), n)).tolist()
    meet = intersect(equations_for_span(gU), equations_for_span(gW), n=n)
    # dim(U ∩ W) = dim U + dim W − dim(U + W)
    assert len(meet) == rank(gU) + rank(gW) - dim_sum(gU, gW)


def test_intersect_with_no_equations_is_everything():
    assert intersect([], [], n=2) == [[1, 0], [0, 1]]
    with pytest.raises(ValueError):
        intersect([], [])


# --- pivot read-off --------------------------------------------------------------------

def test_pivot_report_reads_rank_and_spare_rows():
    text = pivot_report([[1, 2, 0, 3, 1], [2, 4, 1, 7, 1], [1, 2, 1, 4, 0]])
    assert "row 1: first non-zero in column x1" in text
    assert "row 2: first non-zero in column x3" in text
    assert "row 3: all zeros -> a spare equation" in text
    assert "free columns: x2, x4, x5" in text
    assert "= 5 - 2 = 3" in text


# --- data in a subspace / PCA preview -------------------------------------------------

def test_data_on_an_m_dim_subspace_has_n_minus_m_zero_variances():
    basis = rng.normal(size=(2, 5))
    X = rng.normal(size=(100, 2)) @ basis
    v = component_variances(X)
    assert np.sum(v > 1e-9) == 2
    assert np.allclose(v[2:], 0, atol=1e-9)
    assert np.linalg.matrix_rank(X) == 2


def test_noise_makes_them_tiny_not_zero():
    X = rng.normal(size=(300, 2)) @ rng.normal(size=(2, 4))
    v = component_variances(X + rng.normal(scale=0.01, size=X.shape))
    assert np.all(v[2:] > 0)
    assert np.all(v[2:] < 1e-3)
    assert np.all(v[:2] > 0.1)


def test_skipping_the_mean_adds_a_fake_dimension_for_a_shifted_plane():
    X = rng.normal(size=(200, 2)) @ rng.normal(size=(2, 4)) + [5.0, -3.0, 2.0, 4.0]
    assert np.sum(component_variances(X) > 1e-9) == 2
    assert np.sum(component_variances(X, center=False) > 1e-9) == 3
