"""Cofactor expansion, Cramer's rule and areas from corners, with operation counts.

    det_cofactor(A)            expand along the row or column with the most zeros
    det_lu(A)                  the elimination route (det_area.det), with a counter
    cramer(A, b)               x_i = det(A_i) / det(A), A_i = A with column i set to b
    parallelogram_area(c)      four corners in order around the shape -> |det|
    parallelepiped_volume(o, a, b, c)   three edges from one corner -> |det|

Every function takes an optional `counter` (from ../gaussian-elimination/triangular.py)
that counts multiplications and divisions, so "cofactors grow like n!" can be measured
instead of just stated. Exact fractions throughout.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from fractions import Fraction
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gaussian-elimination"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from det_area import permutation_sign  # noqa: E402
from gauss import Matrix, Number, to_matrix  # noqa: E402
from triangular import Counter, lu  # noqa: E402


def _square(A: Sequence[Sequence[Number]]) -> Matrix:
    M = to_matrix(A)
    if any(len(row) != len(M) for row in M):
        raise ValueError("determinants need a square matrix")
    return M


def minor(M: Matrix, i: int, j: int) -> Matrix:
    """M without row i and column j."""
    return [row[:j] + row[j + 1:] for r, row in enumerate(M) if r != i]


def det_cofactor(A: Sequence[Sequence[Number]], counter: Counter | None = None) -> Fraction:
    """Cofactor (Laplace) expansion, the way you'd do it by hand.

    Expand along whichever row or column has the most zeros, since a 0 entry kills its
    whole term. Signs follow the checkerboard + − + ...: (−1)^(i + j).
    """
    M = _square(A)
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        if counter:
            counter.tick(2)
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]

    rows = [sum(x == 0 for x in M[i]) for i in range(n)]
    cols = [sum(M[i][j] == 0 for i in range(n)) for j in range(n)]
    best_row, best_col = max(range(n), key=rows.__getitem__), max(range(n), key=cols.__getitem__)
    if cols[best_col] > rows[best_row]:
        line = [(i, best_col) for i in range(n)]  # down a column
    else:
        line = [(best_row, j) for j in range(n)]  # along a row

    total = Fraction(0)
    for i, j in line:
        if M[i][j] == 0:
            continue
        if counter:
            counter.tick()  # entry × its minor
        sign = 1 if (i + j) % 2 == 0 else -1
        total += sign * M[i][j] * det_cofactor(minor(M, i, j), counter)
    return total


def det_lu(A: Sequence[Sequence[Number]], counter: Counter | None = None) -> Fraction:
    """Elimination: P A = L U, det = sign(P) × product of U's pivots."""
    M = _square(A)
    try:
        f = lu(M, counter)
    except ZeroDivisionError:
        return Fraction(0)
    if counter:
        counter.tick(len(M) - 1)  # multiplying the pivots together
    return permutation_sign(f.perm) * prod((f.U[i][i] for i in range(len(M))), start=Fraction(1))


def cramer(A: Sequence[Sequence[Number]], b: Sequence[Number],
           counter: Counter | None = None, method=det_lu) -> list[Fraction]:
    """x_i = det(A_i) / det(A), where A_i is A with column i replaced by b."""
    M = _square(A)
    b = [Fraction(v) for v in b]
    if len(b) != len(M):
        raise ValueError("b needs one entry per row of A")
    D = method(M, counter)
    if D == 0:
        raise ZeroDivisionError("det A = 0: A is singular, nothing to divide by")
    x = []
    for i in range(len(M)):
        Ai = [row[:i] + [bk] + row[i + 1:] for row, bk in zip(M, b, strict=True)]
        x.append(method(Ai, counter) / D)
        if counter:
            counter.tick()  # the division
    return x


def parallelogram_area(corners: Sequence[Sequence[Number]]) -> Fraction:
    """Area from four corners listed in order around the shape.

    Edges from the first corner go to its two neighbours (corners 1 and 3); corner 2
    must be where those edges meet, or the shape isn't a parallelogram.
    """
    if len(corners) != 4:
        raise ValueError("a parallelogram has 4 corners")
    p0, p1, p2, p3 = ([Fraction(v) for v in c] for c in corners)
    u = [a - o for a, o in zip(p1, p0, strict=True)]
    w = [a - o for a, o in zip(p3, p0, strict=True)]
    if [o + x + y for o, x, y in zip(p0, u, w, strict=True)] != p2:
        raise ValueError("not a parallelogram in this order (opposite corner doesn't match)")
    return abs(u[0] * w[1] - w[0] * u[1])


def parallelepiped_volume(origin, a, b, c) -> Fraction:
    """Volume of the 3-D box-like solid spanned by three edges from one corner."""
    edges = [[Fraction(p) - Fraction(o) for p, o in zip(q, origin, strict=True)] for q in (a, b, c)]
    columns = [[edges[k][r] for k in range(3)] for r in range(3)]  # edges as columns
    return abs(det_lu(columns))


def operation_counts(n: int, seed: int = 0) -> dict[str, int]:
    """Multiplications + divisions for one dense n×n determinant and one Cramer solve."""
    import random

    rnd = random.Random(seed)
    A = [[rnd.randint(1, 9) for _ in range(n)] for _ in range(n)]  # no zeros to skip
    b = [rnd.randint(1, 9) for _ in range(n)]
    counts = {}
    for name, fn in [("cofactor", det_cofactor), ("elimination", det_lu)]:
        c = Counter()
        fn(A, c)
        counts[name] = c.ops
    c = Counter()
    cramer(A, b, c)
    counts["cramer"] = c.ops  # n + 1 elimination determinants
    return counts


if __name__ == "__main__":
    A = [[2, 0, 1, 3], [1, 0, 0, 2], [4, 5, 1, 0], [0, 0, 3, 1]]
    print("A =", A)
    print(f"  cofactor (down column 2, three zeros): {det_cofactor(A)}")
    print(f"  elimination:                          {det_lu(A)}")

    x = cramer([[4, 1], [2, 5]], [9, 21])
    print("\nCramer on [[4, 1], [2, 5]] x = [9, 21]:", [str(v) for v in x])
    corners = [(2, 1), (7, 2), (8, 6), (3, 5)]
    print("parallelogram (2,1),(7,2),(8,6),(3,5):", parallelogram_area(corners))
    print("parallelepiped edges (2,0,0),(1,3,0),(0,1,4):",
          parallelepiped_volume((0, 0, 0), (2, 0, 0), (1, 3, 0), (0, 1, 4)))

    print("\nmultiplications + divisions for one dense n x n problem")
    print(" n   cofactor   elimination   cramer")
    for n in range(2, 9):
        c = operation_counts(n)
        print(f"{n:2d} {c['cofactor']:10,d} {c['elimination']:13,d} {c['cramer']:8,d}")
