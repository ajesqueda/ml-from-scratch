"""Matrix times vector, read two ways. Both always give the same answer.

    row view      answer[i] = (row i of A) . x
                  How you compute by hand, one dot product per output entry.

    column view   answer = x[0]*(column 0) + x[1]*(column 1) + ...
                  x is a recipe for mixing A's columns. This is the view behind
                  Ax = b: "which mix of A's columns produces b?"

Shapes: an m x n matrix takes a vector with n entries and returns m entries.
"""

from __future__ import annotations

from collections.abc import Sequence

Number = int | float
Vector = Sequence[Number]


def _check_shapes(A: Sequence[Vector], x: Vector) -> int:
    n = len(x)
    for i, row in enumerate(A):
        if len(row) != n:
            raise ValueError(
                f"shape mismatch: row {i} of A has {len(row)} entries, so x needs "
                f"{len(row)} entries, but it has {n} (an m x n matrix takes n entries)"
            )
    return n


def matvec_rows(A: Sequence[Vector], x: Vector) -> list[Number]:
    """Row view: one dot product per row of A."""
    _check_shapes(A, x)
    return [sum(a * xi for a, xi in zip(row, x, strict=True)) for row in A]


def matvec_columns(A: Sequence[Vector], x: Vector) -> list[Number]:
    """Column view: add up A's columns, each scaled by its entry of x."""
    n = _check_shapes(A, x)
    result: list[Number] = [0] * len(A)
    for j in range(n):
        column = [row[j] for row in A]
        result = [r + x[j] * c for r, c in zip(result, column, strict=True)]
    return result


if __name__ == "__main__":
    A = [[2, 1], [1, 3]]
    x = [3, 2]
    print("row view:   ", matvec_rows(A, x), "  = [2*3 + 1*2, 1*3 + 3*2]")
    print("column view:", matvec_columns(A, x), "  = 3*[2, 1] + 2*[1, 3]")
