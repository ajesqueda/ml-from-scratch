from fractions import Fraction

import numpy as np
import pytest
from gauss import Eliminator, rank, to_matrix
from matvec import matvec_columns, matvec_rows


def test_example_from_notes():
    A, x = [[2, 1], [1, 3]], [3, 2]
    assert matvec_rows(A, x) == matvec_columns(A, x) == [8, 9]


@pytest.mark.parametrize("seed", range(30))
def test_row_and_column_views_agree_with_numpy(seed):
    rng = np.random.default_rng(seed)
    m, n = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    A = rng.integers(-9, 10, size=(m, n))
    x = rng.integers(-9, 10, size=n)
    expected = (A @ x).tolist()
    assert matvec_rows(A.tolist(), x.tolist()) == expected
    assert matvec_columns(A.tolist(), x.tolist()) == expected
    assert len(expected) == m  # m x n times n entries gives m entries


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="x needs 3 entries, but it has 2"):
        matvec_rows([[1, 2, 3]], [1, 2])
    with pytest.raises(ValueError):
        matvec_columns([[1, 2], [3, 4]], [1, 2, 3])


# --- every row operation can be undone ------------------------------------


def apply(e: Eliminator, op: tuple) -> None:
    kind, *args = op
    getattr(e, kind)(*args)


def inverse_op(op: tuple) -> tuple:
    kind, *args = op
    if kind == "swap":
        return op  # swapping twice puts the rows back
    if kind == "scale":
        i, factor = args
        return ("scale", i, 1 / factor)
    target, source, multiplier = args  # R_t <- R_t - m*R_s is undone by + m*R_s
    return ("add_multiple", target, source, -multiplier)


def random_ops(rng: np.random.Generator, n_rows: int, count: int) -> list[tuple]:
    ops = []
    for _ in range(count):
        i, j = (int(v) for v in rng.choice(n_rows, size=2, replace=False))
        k = int(rng.integers(3))
        value = Fraction(int(rng.choice([-3, -2, -1, 1, 2, 3])), int(rng.integers(1, 4)))
        ops.append([("swap", i, j), ("scale", i, value), ("add_multiple", i, j, value)][k])
    return ops


@pytest.mark.parametrize("seed", range(25))
def test_undoing_moves_in_reverse_order_restores_the_matrix(seed):
    rng = np.random.default_rng(500 + seed)
    rows, cols = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    original = to_matrix(rng.integers(-5, 6, size=(rows, cols)).tolist())
    ops = random_ops(rng, rows, count=8)

    e = Eliminator([row[:] for row in original])
    for op in ops:
        apply(e, op)
        assert rank(e.matrix) == rank(original)  # legal moves never change rank
    for op in reversed(ops):
        apply(e, inverse_op(op))
    assert e.matrix == original


def test_undoing_in_the_wrong_order_usually_fails():
    # Order matters: R1 <- R1 - R2, then swap. Undoing the add before the swap
    # touches the wrong row.
    original = to_matrix([[1, 2], [3, 4]])
    ops = [("add_multiple", 0, 1, Fraction(1)), ("swap", 0, 1)]
    e = Eliminator([row[:] for row in original])
    for op in ops:
        apply(e, op)
    for op in ops:  # forward order instead of reversed
        apply(e, inverse_op(op))
    assert e.matrix != original


def test_scale_by_zero_is_not_a_legal_move():
    with pytest.raises(ValueError, match="not a legal row operation"):
        Eliminator(to_matrix([[1, 2]])).scale(0, Fraction(0))
