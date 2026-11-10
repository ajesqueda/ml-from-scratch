from fractions import Fraction

import numpy as np
import pytest
from det_area import (
    box_proof,
    det,
    det_2x2,
    permutation_sign,
    signed_area,
    transform,
)
from gauss import inverse, rank

rng = np.random.default_rng(611)


def random_int(n, low=-5, high=6):
    return rng.integers(low, high, size=(n, n)).tolist()


def matmul(A, B):
    cols = list(zip(*B, strict=True))
    return [[sum(Fraction(a) * b for a, b in zip(row, col, strict=True)) for col in cols]
            for row in A]


# --- det from LU ----------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_det_matches_numpy(n):
    for _ in range(20):
        A = random_int(n)
        assert float(det(A)) == pytest.approx(np.linalg.det(A), abs=1e-6)


def test_det_is_exact_where_floating_point_is_not():
    B = [[0, 2, 1], [1, 1, 0], [2, 0, 3]]
    assert det(B) == -8
    assert np.linalg.det(B) != -8  # -7.999999999999998


def test_2x2_formula_agrees_with_lu():
    for _ in range(50):
        A = random_int(2)
        assert det(A) == det_2x2(A)


def test_triangular_det_is_the_diagonal_product():
    U = [[2, 7, -1], [0, -3, 4], [0, 0, 5]]
    assert det(U) == -30
    assert det([[row[i] for row in U] for i in range(3)]) == -30  # det(Uᵀ) too


@pytest.mark.parametrize("perm, sign", [
    ([0, 1, 2], 1), ([1, 0, 2], -1), ([1, 2, 0], 1), ([2, 1, 0], -1), ([3, 2, 1, 0], 1),
])
def test_permutation_sign(perm, sign):
    assert permutation_sign(perm) == sign


# --- singular <=> det 0 <=> rank drops <=> no inverse -------------------------------

def test_singular_matrices_have_det_exactly_zero():
    for A in ([[2, 4], [1, 2]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0], [0, 0]]):
        assert det(A) == 0
        assert rank(A) < len(A)
        with pytest.raises(ValueError, match="singular"):
            inverse(A)


def test_nonzero_det_means_full_rank():
    for _ in range(30):
        A = random_int(3)
        assert (det(A) != 0) == (rank(A) == 3)


# --- rules that fall out of the LU view -------------------------------------------

def test_row_swap_flips_the_sign():
    A = random_int(4)
    swapped = [A[2], A[1], A[0], A[3]]
    assert det(swapped) == -det(A)


def test_adding_a_multiple_of_another_row_changes_nothing():
    A = random_int(3)
    B = [A[0], [x + 3 * y for x, y in zip(A[1], A[0], strict=True)], A[2]]
    assert det(B) == det(A)


def test_scaling_one_row_scales_det():
    A = random_int(3)
    B = [A[0], [-2 * x for x in A[1]], A[2]]
    assert det(B) == -2 * det(A)


def test_product_rule_and_inverse():
    for _ in range(10):
        A, B = random_int(3), random_int(3)
        assert det(matmul(A, B)) == det(A) * det(B)
        if det(A) != 0:
            assert det(inverse(A)) == 1 / det(A)


# --- area -------------------------------------------------------------------------------

def test_box_proof_matches_ad_minus_bc():
    assert box_proof(3, 1, 1, 2)["area"] == 5
    for a, b, c, d in rng.integers(-6, 7, size=(50, 4)).tolist():
        assert box_proof(a, b, c, d)["area"] == a * d - b * c


def test_unit_square_lands_on_the_columns():
    A = [[3, 1], [1, 2]]
    image = transform(A, [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert image[1] == (3, 1) and image[3] == (1, 2)  # column 1 and column 2
    assert signed_area(image) == det(A)


def test_shoelace_orientation():
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert signed_area(square) == 4
    assert signed_area(square[::-1]) == -4


def random_polygon(k):
    """A simple (non-crossing) polygon: random radii at sorted angles around a center."""
    angles = np.sort(rng.uniform(0, 2 * np.pi, k))
    radii = rng.uniform(1, 5, k)
    return [(Fraction(round(r * np.cos(t), 3)), Fraction(round(r * np.sin(t), 3)))
            for r, t in zip(radii, angles, strict=True)]


@pytest.mark.parametrize("k", [3, 5, 8])
def test_any_shape_scales_by_det(k):
    for _ in range(10):
        A = random_int(2)
        poly = random_polygon(k)
        assert signed_area(transform(A, poly)) == det(A) * signed_area(poly)
        assert abs(signed_area(transform(A, poly))) == abs(det(A)) * abs(signed_area(poly))


def test_negative_det_mirrors_the_shape():
    mirror = [[0, 1], [1, 0]]  # swap x and y
    triangle = [(0, 0), (3, 0), (0, 1)]
    assert det(mirror) == -1
    assert signed_area(triangle) > 0 > signed_area(transform(mirror, triangle))


# --- inverse rows are recipes ------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4])
def test_each_inverse_row_isolates_one_unknown(n):
    A = random_int(n)
    while det(A) == 0:
        A = random_int(n)
    A_inv = inverse(A)
    # Row i of A⁻¹ times column j of A is 1 when i == j and 0 otherwise...
    assert matmul(A_inv, A) == [[int(i == j) for j in range(n)] for i in range(n)]
    # ...so dotting row i with b = A x picks out exactly x_i.
    x = rng.integers(-9, 10, size=n).tolist()
    b = [sum(Fraction(a) * v for a, v in zip(row, x, strict=True)) for row in A]
    for i in range(n):
        assert sum(r * v for r, v in zip(A_inv[i], b, strict=True)) == x[i]
