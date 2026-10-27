from fractions import Fraction

import numpy as np
import pytest
import sympy
from gauss import inverse, rank, row_echelon, rref, solve_system, to_matrix


def random_int_matrix(rng: np.random.Generator, rows: int, cols: int, rank_cap: int | None = None):
    """Small integer matrix; with rank_cap, built as a product so rank <= rank_cap."""
    if rank_cap is None:
        return rng.integers(-5, 6, size=(rows, cols)).tolist()
    left = rng.integers(-3, 4, size=(rows, rank_cap))
    right = rng.integers(-3, 4, size=(rank_cap, cols))
    return (left @ right).tolist()


def sympy_rref(rows) -> list[list[Fraction]]:
    reduced, _ = sympy.Matrix(rows).rref()
    return [[Fraction(int(x.p), int(x.q)) for x in reduced.row(i)] for i in range(reduced.rows)]


# --- rank and RREF --------------------------------------------------------


def test_rank_example_from_notes():
    assert rank([[1, 1, 2], [2, 3, 5], [1, 2, 3]]) == 2
    assert rank([[1, 6, 4], [2, 0, 3], [3, 6, 7]]) == 2


@pytest.mark.parametrize("seed", range(40))
def test_rank_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(1, 6)), int(rng.integers(1, 6)))
    cap = int(rng.integers(1, min(shape) + 1)) if seed % 2 else None
    m = random_int_matrix(rng, *shape, rank_cap=cap)
    assert rank(m) == np.linalg.matrix_rank(np.array(m, dtype=float))


@pytest.mark.parametrize("seed", range(40))
def test_rref_matches_sympy_exactly(seed):
    rng = np.random.default_rng(1000 + seed)
    shape = (int(rng.integers(1, 6)), int(rng.integers(1, 7)))
    m = random_int_matrix(rng, *shape, rank_cap=int(rng.integers(1, min(shape) + 1)))
    reduced, _ = rref(m)
    assert reduced == sympy_rref(m)


def test_forward_pass_gives_echelon_form():
    ref, pivots = row_echelon([[1, 6, 4], [2, 0, 3], [3, 6, 7]])
    assert pivots == [0, 1]
    assert ref[2] == [0, 0, 0]  # dependent row collapses to zeros
    assert all(ref[r][c] == 0 for r in range(1, 3) for c in range(r) if c < len(ref[0]))


def test_zero_pivot_forces_a_row_swap():
    _, pivots = rref([[0, 1], [1, 0]])
    assert pivots == [0, 1]


def test_row_operations_preserve_fractions():
    reduced, _ = rref([[2, 3], [4, 9]])
    assert reduced == to_matrix([[1, 0], [0, 1]])
    # Multiplier 3/2 (non-unit pivot) stays exact rather than 1.4999...
    ref, _ = row_echelon([[2, 1], [3, 1]])
    assert ref[1] == [0, Fraction(-1, 2)]


# --- solving Ax = b: none / unique / infinite ----------------------------


def test_unique_solution_from_notes():
    sol = solve_system([[1, 2], [3, 4]], [5, 11])
    assert sol.kind == "unique"
    assert sol.particular == [1, 2]


@pytest.mark.parametrize("seed", range(25))
def test_unique_solutions_match_numpy(seed):
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(1, 6))
    A = random_int_matrix(rng, n, n)
    if np.linalg.matrix_rank(np.array(A, dtype=float)) < n:
        pytest.skip("random matrix happened to be singular")
    b = rng.integers(-10, 11, size=n).tolist()
    sol = solve_system(A, b)
    assert sol.kind == "unique"
    np.testing.assert_allclose([float(x) for x in sol.particular], np.linalg.solve(A, b))


def test_inconsistent_system_has_no_solution():
    # Row 3 = row 1 + row 2 on the left, but 3 + 8 != 12 on the right.
    sol = solve_system([[1, 1, 2], [2, 3, 5], [3, 4, 7]], [3, 8, 12])
    assert sol.kind == "none"


def test_dependent_system_gives_parametric_solution():
    A = [[1, 1, 2], [2, 3, 5], [3, 4, 7]]
    b = [3, 8, 11]
    sol = solve_system(A, b)
    assert sol.kind == "infinite"
    assert sol.free_vars == [2] and sol.bound_vars == [0, 1]
    assert sol.particular == [1, 2, 0]
    assert sol.directions == [[-1, -1, 1]]


@pytest.mark.parametrize("seed", range(25))
def test_every_parametric_point_solves_the_system(seed):
    rng = np.random.default_rng(3000 + seed)
    rows, cols = int(rng.integers(2, 5)), int(rng.integers(3, 6))
    A = random_int_matrix(rng, rows, cols, rank_cap=int(rng.integers(1, min(rows, cols))))
    x_true = rng.integers(-4, 5, size=cols)
    b = (np.array(A) @ x_true).tolist()  # consistent by construction
    sol = solve_system(A, b)
    assert sol.kind == "infinite"
    assert len(sol.free_vars) == cols - rank(A)
    a = to_matrix(A)
    for ts in rng.integers(-5, 6, size=(5, len(sol.directions))):
        x = list(sol.particular)
        for t, d in zip(ts, sol.directions, strict=True):
            x = [xi + int(t) * di for xi, di in zip(x, d, strict=True)]
        assert [sum(r * xi for r, xi in zip(row, x, strict=True)) for row in a] == b


def test_more_unknowns_than_equations():
    sol = solve_system([[1, 1, 1]], [6])
    assert sol.kind == "infinite"
    assert sol.free_vars == [1, 2]


def test_b_length_is_checked():
    with pytest.raises(ValueError):
        solve_system([[1, 0], [0, 1]], [1])


# --- inverse via [A | I] ---------------------------------------------------


def test_inverse_from_notes():
    assert inverse([[1, 1], [2, 3]]) == to_matrix([[3, -1], [-2, 1]])


@pytest.mark.parametrize("seed", range(20))
def test_inverse_matches_numpy(seed):
    rng = np.random.default_rng(4000 + seed)
    n = int(rng.integers(1, 5))
    A = random_int_matrix(rng, n, n)
    if np.linalg.matrix_rank(np.array(A, dtype=float)) < n:
        pytest.skip("random matrix happened to be singular")
    inv = np.array([[float(x) for x in row] for row in inverse(A)])
    np.testing.assert_allclose(inv, np.linalg.inv(A), atol=1e-12)


def test_singular_matrix_has_no_inverse():
    with pytest.raises(ValueError, match="singular"):
        inverse([[1, 2], [2, 4]])


def test_trace_records_each_row_operation(capsys):
    solve_system([[0, 2], [1, 1]], [4, 3], trace=True)
    out = capsys.readouterr().out
    assert out.startswith("R1 <-> R2")
    assert "R2 <- (1/2)*R2" in out
