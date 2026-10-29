from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg
from gauss import solve_system, to_matrix
from triangular import (
    Counter,
    back_sub,
    classify_by_rank,
    forward_sub,
    is_positive_definite,
    lu,
    lu_solve,
    quadratic_minimum,
)


def as_float(M):
    return np.array([[float(v) for v in row] for row in M])


def random_square(rng, n, low=-6, high=7):
    while True:
        A = rng.integers(low, high, size=(n, n))
        if np.linalg.matrix_rank(A) == n:
            return A.tolist()


# --- triangular solves ----------------------------------------------------


def test_examples_from_notes():
    assert forward_sub([[3, 0, 0], [2, 1, 0], [1, -2, 4]], [6, 7, 4]) == [2, 3, 2]
    assert back_sub([[2, 1, 1], [0, 1, -1], [0, 0, 2]], [9, -1, 4]) == [3, 1, 2]


@pytest.mark.parametrize("seed", range(20))
def test_triangular_solves_match_scipy(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 7))
    full = np.array(random_square(rng, n), dtype=float)
    lower = np.tril(full) + np.eye(n) * 10  # keep the diagonal non-zero
    upper = np.triu(full) + np.eye(n) * 10
    b = rng.integers(-9, 10, size=n)
    np.testing.assert_allclose(
        [float(v) for v in forward_sub(lower.tolist(), b.tolist())],
        scipy.linalg.solve_triangular(lower, b, lower=True),
    )
    np.testing.assert_allclose(
        [float(v) for v in back_sub(upper.tolist(), b.tolist())],
        scipy.linalg.solve_triangular(upper, b, lower=False),
    )


def test_zero_on_the_diagonal_explains_itself():
    with pytest.raises(ZeroDivisionError, match="can't be pinned down"):
        back_sub([[1, 2], [0, 0]], [3, 0])


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="lower-triangular"):
        forward_sub([[1, 2], [0, 1]], [1, 1])
    with pytest.raises(ValueError, match="upper-triangular"):
        back_sub([[1, 0], [2, 1]], [1, 1])
    with pytest.raises(ValueError, match="b has 1 entries"):
        back_sub([[1, 0], [0, 1]], [1])


# --- LU -------------------------------------------------------------------


def multiply(A, B):
    return [[sum(a * b for a, b in zip(row, col, strict=True)) for col in zip(*B, strict=True)]
            for row in A]


@pytest.mark.parametrize("seed", range(25))
def test_lu_rebuilds_the_permuted_matrix_exactly(seed):
    rng = np.random.default_rng(100 + seed)
    A = random_square(rng, int(rng.integers(1, 6)))
    f = lu(A)
    a = to_matrix(A)
    assert multiply(f.L, f.U) == [a[i] for i in f.perm]  # P A = L U, fraction for fraction
    n = len(A)
    assert all(f.L[i][i] == 1 and all(f.L[i][j] == 0 for j in range(i + 1, n)) for i in range(n))
    assert all(f.U[i][j] == 0 for i in range(n) for j in range(i))


@pytest.mark.parametrize("seed", range(15))
def test_lu_agrees_with_scipy(seed):
    # SciPy uses partial pivoting (largest pivot), so its L and U differ from
    # ours; compare what both must agree on: the determinant and the solution.
    rng = np.random.default_rng(200 + seed)
    A = random_square(rng, int(rng.integers(2, 6)))
    f = lu(A)
    sign = np.linalg.det(np.eye(len(A))[f.perm])
    det_ours = sign * np.prod([float(f.U[i][i]) for i in range(len(A))])
    P, L, U = scipy.linalg.lu(np.array(A, dtype=float))
    assert det_ours == pytest.approx(np.linalg.det(P) * np.prod(np.diag(U)))
    b = rng.integers(-9, 10, size=len(A))
    np.testing.assert_allclose(
        [float(v) for v in lu_solve(f, b.tolist())], scipy.linalg.solve(np.array(A, float), b)
    )


def test_zero_pivot_triggers_a_row_swap():
    f = lu([[0, 1], [2, 3]])
    assert f.perm == [1, 0]
    assert lu_solve(f, [1, 5]) == [1, 1]


def test_singular_matrix_has_no_lu():
    with pytest.raises(ZeroDivisionError, match="singular"):
        lu([[1, 2], [2, 4]])


def test_factor_once_solve_many():
    A = [[4, -2, 1], [-2, 4, -2], [1, -2, 4]]
    f = lu(A)
    for b in ([11, -16, 17], [1, 0, 0], [0, 0, 1]):
        x = lu_solve(f, b)
        assert solve_system(A, b).particular == x


# --- cost: n³ vs n² -------------------------------------------------------


def count(fn, *args):
    c = Counter()
    result = fn(*args, c)
    return c.ops, result


def test_triangular_solve_costs_n_squared_over_two():
    for n in (5, 10, 20):
        ops, _ = count(back_sub, np.triu(np.ones((n, n))).tolist(), [1] * n)
        assert ops == n * (n + 1) // 2


def test_doubling_n_costs_8x_to_factor_and_4x_to_solve():
    def costs(n):
        A = [[n + 1 if i == j else 1 for j in range(n)] for i in range(n)]
        factor_ops, f = count(lu, A)
        solve_ops, _ = count(lu_solve, f, [1] * n)
        return factor_ops, solve_ops

    (f40, s40), (f80, s80) = costs(40), costs(80)
    assert 7.5 < f80 / f40 < 8.5
    assert 3.8 < s80 / s40 < 4.2
    assert f80 > 25 * s80  # factoring dominates; that's why you reuse it


# --- 0, 1 or infinitely many, by rank -------------------------------------


@pytest.mark.parametrize(
    "A,b,expected",
    [
        ([[1, -1], [3, -3]], [2, 5], "none"),
        ([[1, 2], [3, 4]], [5, 11], "unique"),
        ([[1, 1, 1], [1, 2, 3], [2, 3, 4]], [3, 6, 9], "infinite"),
    ],
)
def test_classify_examples_from_notes(A, b, expected):
    assert classify_by_rank(A, b) == expected


@pytest.mark.parametrize("seed", range(40))
def test_rank_rule_agrees_with_elimination(seed):
    rng = np.random.default_rng(300 + seed)
    rows, cols = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    r = int(rng.integers(1, min(rows, cols) + 1))
    A = (rng.integers(-3, 4, size=(rows, r)) @ rng.integers(-3, 4, size=(r, cols))).tolist()
    kind = seed % 3
    if kind == 0:  # consistent by construction
        b = (np.array(A) @ rng.integers(-3, 4, size=cols)).tolist()
    else:  # random b: often inconsistent when A is short of full rank
        b = rng.integers(-5, 6, size=rows).tolist()
    assert classify_by_rank(A, b) == solve_system(A, b).kind


def test_collinear_vs_parallel():
    # Same line: every ratio matches, right side included.
    assert classify_by_rank([[3, 2], [6, 4]], [5, 10]) == "infinite"
    # Parallel: left ratios match (2), right side doesn't (9/4).
    assert classify_by_rank([[2, 3], [4, 6]], [4, 9]) == "none"


# --- quadratic minimum = linear system ------------------------------------


def test_quadratic_minimum_from_notes():
    assert quadratic_minimum([[2, 1], [1, 2]], [-3, -3]) == [1, 1]


def test_minimum_beats_every_grid_point():
    H, g = [[4, 1, 0], [1, 3, 1], [0, 1, 2]], [-2, 5, -1]
    x_star = np.array([float(v) for v in quadratic_minimum(H, g)])
    Hf, gf = np.array(H, float), np.array(g, float)

    def f(x):
        return 0.5 * x @ Hf @ x + gf @ x

    grid = np.linspace(-3, 3, 13)
    best_grid = min(f(np.array([a, b, c])) for a in grid for b in grid for c in grid)
    assert f(x_star) <= best_grid + 1e-12
    np.testing.assert_allclose(Hf @ x_star + gf, 0, atol=1e-12)  # gradient is zero


def test_saddle_is_not_a_minimum():
    # f = x² - y² has gradient 0 at the origin, but it's a saddle, not a minimum.
    assert not is_positive_definite([[2, 0], [0, -2]])
    with pytest.raises(ValueError, match="positive definite"):
        quadratic_minimum([[2, 0], [0, -2]], [0, 0])


def test_positive_definite_checks():
    assert is_positive_definite([[2, 1], [1, 2]])
    assert not is_positive_definite([[1, 2], [3, 4]])  # not symmetric
    assert not is_positive_definite([[1, 2], [2, 4]])  # singular
    assert Fraction(1, 2) in lu([[2, 1], [1, 2]]).L[1]  # multiplier 1/2 kept exactly
