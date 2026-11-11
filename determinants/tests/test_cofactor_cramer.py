from fractions import Fraction

import numpy as np
import pytest
from cofactor_cramer import (
    cramer,
    det_cofactor,
    det_lu,
    minor,
    operation_counts,
    parallelepiped_volume,
    parallelogram_area,
)
from gauss import inverse, rank, to_matrix
from subspaces import null_space
from triangular import Counter, lu, lu_solve

rng = np.random.default_rng(1110)


def random_int(n, low=-5, high=6):
    return rng.integers(low, high, size=(n, n)).tolist()


def transpose(A):
    return [list(col) for col in zip(*A, strict=True)]


# --- cofactor expansion -------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_cofactor_matches_elimination_and_numpy(n):
    for _ in range(15):
        A = random_int(n)
        assert det_cofactor(A) == det_lu(A)
        assert float(det_cofactor(A)) == pytest.approx(np.linalg.det(A), abs=1e-6)


def test_minor_drops_one_row_and_one_column():
    M = to_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert minor(M, 1, 0) == [[2, 3], [8, 9]]


def test_expanding_along_zeros_does_less_work():
    sparse = [[2, 0, 1, 3], [1, 0, 0, 2], [4, 5, 1, 0], [0, 0, 3, 1]]
    dense = [[2, 6, 1, 3], [1, 7, 4, 2], [4, 5, 1, 8], [9, 2, 3, 1]]
    c_sparse, c_dense = Counter(), Counter()
    assert det_cofactor(sparse, c_sparse) == 20
    det_cofactor(dense, c_dense)
    assert c_sparse.ops < c_dense.ops / 4


def test_non_square_is_rejected():
    with pytest.raises(ValueError, match="square"):
        det_cofactor([[1, 2, 3], [4, 5, 6]])


# --- more rules ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4])
def test_transpose_keeps_det(n):
    A = random_int(n)
    assert det_lu(transpose(A)) == det_lu(A)


@pytest.mark.parametrize("n, c", [(2, 3), (3, 2), (3, -1), (4, Fraction(1, 2))])
def test_scaling_the_whole_matrix_scales_det_by_c_to_the_n(n, c):
    A = random_int(n)
    assert det_lu([[c * x for x in row] for row in A]) == c**n * det_lu(A)


def test_det_of_c_times_inverse_is_c_n_over_det():
    A = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]  # det 18
    cA_inv = [[2 * x for x in row] for row in inverse(A)]
    assert det_lu(A) == 18
    assert det_lu(cA_inv) == Fraction(2**3, 18)


# --- one fact, four faces ---------------------------------------------------------------

def singular(n):
    A = random_int(n)
    A[-1] = [2 * x - y for x, y in zip(A[0], A[1], strict=True)]  # last row = 2·row0 − row1
    return A


@pytest.mark.parametrize("make", [random_int, singular])
def test_det_zero_rank_null_space_and_inverse_agree(make):
    for _ in range(10):
        A = make(4)
        is_singular = det_lu(A) == 0
        assert is_singular == (rank(A) < 4)
        assert is_singular == bool(null_space(A))  # a non-zero x with Ax = 0
        try:
            inverse(A)
            has_inverse = True
        except ValueError:
            has_inverse = False
        assert is_singular == (not has_inverse)


def test_singular_in_t():
    # For which t is [[1, t], [2, 6]] singular? det = 6 − 2t = 0 -> t = 3
    assert det_lu([[1, 3], [2, 6]]) == 0
    assert det_lu([[1, 2], [2, 6]]) != 0


# --- Cramer's rule -------------------------------------------------------------------------

def test_cramer_small_example():
    assert cramer([[4, 1], [2, 5]], [9, 21]) == [Fraction(4, 3), Fraction(11, 3)]


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_cramer_matches_lu_solve(n):
    for _ in range(10):
        A = random_int(n)
        if det_lu(A) == 0:
            continue
        b = rng.integers(-9, 10, size=n).tolist()
        assert cramer(A, b) == lu_solve(lu(A), b)
        assert cramer(A, b, method=det_cofactor) == lu_solve(lu(A), b)


def test_cramer_refuses_singular():
    with pytest.raises(ZeroDivisionError, match="singular"):
        cramer([[1, 2], [2, 4]], [3, 6])


# --- areas and volumes ----------------------------------------------------------------------

def test_parallelogram_from_corners():
    assert parallelogram_area([(2, 1), (7, 2), (8, 6), (3, 5)]) == 19
    assert parallelogram_area([(3, 5), (8, 6), (7, 2), (2, 1)]) == 19  # other direction
    with pytest.raises(ValueError, match="parallelogram"):
        parallelogram_area([(0, 0), (4, 0), (5, 3), (0, 2)])


def test_parallelogram_area_ignores_translation():
    base = [(0, 0), (5, 1), (6, 5), (1, 4)]
    shifted = [(x + 10, y - 7) for x, y in base]
    assert parallelogram_area(base) == parallelogram_area(shifted)


def test_parallelepiped_volume():
    assert parallelepiped_volume((0, 0, 0), (2, 0, 0), (1, 3, 0), (0, 1, 4)) == 24
    assert parallelepiped_volume((1, 1, 1), (3, 1, 1), (2, 4, 1), (1, 2, 5)) == 24
    assert parallelepiped_volume((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)) == 0  # flat


# --- the cost story ---------------------------------------------------------------------------

def test_cofactor_grows_factorially_elimination_cubically():
    counts = {n: operation_counts(n) for n in range(3, 9)}
    for n in range(4, 9):
        # each size multiplies cofactor work by about n...
        assert counts[n]["cofactor"] >= (n - 1) * counts[n - 1]["cofactor"]
        # ...while elimination stays within n³
        assert counts[n]["elimination"] <= n**3
    assert counts[8]["cofactor"] > 300 * counts[8]["elimination"]
    assert counts[8]["cramer"] > 8 * counts[8]["elimination"]
