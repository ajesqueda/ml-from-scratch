"""Determinants as area scale factors, computed exactly from an LU factorization.

    det(A)                   product of U's pivots, sign flipped once per row swap
    det_2x2(A)               ad − bc
    box_proof(a, b, c, d)    ad − bc as a box minus its corner pieces (no graph needed)
    signed_area(polygon)     shoelace formula; negative means clockwise
    transform(A, polygon)    move every vertex: v -> A v

Built on the exact-fraction LU in ../gaussian-elimination/triangular.py, so a
singular matrix gives exactly 0, not 1e-16.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from fractions import Fraction
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gaussian-elimination"))
from gauss import Number, to_matrix  # noqa: E402
from triangular import lu  # noqa: E402

Point = tuple[Fraction, Fraction]


def permutation_sign(perm: Sequence[int]) -> int:
    """+1 for an even number of swaps, −1 for odd (count cycles: a k-cycle is k − 1 swaps)."""
    seen = [False] * len(perm)
    swaps = 0
    for start in range(len(perm)):
        length, i = 0, start
        while not seen[i]:
            seen[i] = True
            i = perm[i]
            length += 1
        swaps += max(length - 1, 0)
    return -1 if swaps % 2 else 1


def det(A: Sequence[Sequence[Number]]) -> Fraction:
    """det A from P A = L U: det L = 1 (1s on its diagonal), det U = product of pivots,
    and every row swap in P flips the sign. A matrix with no pivot in some column is
    singular: det = 0."""
    try:
        f = lu(A)
    except ZeroDivisionError:
        return Fraction(0)
    return permutation_sign(f.perm) * prod((f.U[i][i] for i in range(len(f.U))), start=Fraction(1))


def det_2x2(A: Sequence[Sequence[Number]]) -> Fraction:
    (a, b), (c, d) = to_matrix(A)
    return a * d - b * c


def box_proof(a: Number, b: Number, c: Number, d: Number) -> dict[str, Fraction]:
    """Area of the parallelogram spanned by the columns (a, c) and (b, d), without a graph.

    Put it in an (a + b) × (c + d) box. What's left around it is two triangles with
    legs a, c; two with legs b, d; and two b × c rectangles. Drawn for positive entries
    with ad > bc, but the algebra holds for any numbers:
        (a + b)(c + d) − ac − bd − 2bc = ad − bc
    """
    a, b, c, d = (Fraction(x) for x in (a, b, c, d))
    pieces = {
        "box": (a + b) * (c + d),
        "triangles a,c": a * c,      # two triangles, each a·c/2
        "triangles b,d": b * d,      # two triangles, each b·d/2
        "rectangles b,c": 2 * b * c,  # two rectangles, each b·c
    }
    pieces["area"] = (pieces["box"] - pieces["triangles a,c"] - pieces["triangles b,d"]
                      - pieces["rectangles b,c"])
    return pieces


def signed_area(polygon: Sequence[Sequence[Number]]) -> Fraction:
    """Shoelace formula. Positive for counterclockwise vertices, negative for clockwise."""
    pts = [(Fraction(x), Fraction(y)) for x, y in polygon]
    edges = zip(pts, pts[1:] + pts[:1], strict=True)  # each vertex with the next one
    twice = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in edges)
    return twice / 2


def transform(A: Sequence[Sequence[Number]], polygon: Sequence[Sequence[Number]]) -> list[Point]:
    (a, b), (c, d) = to_matrix(A)
    return [(a * x + b * y, c * x + d * y)
            for x, y in ((Fraction(px), Fraction(py)) for px, py in polygon)]


def _s(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


if __name__ == "__main__":
    M = [[3, 1], [1, 2]]
    print("unit square under", M)
    for name, value in box_proof(3, 1, 1, 2).items():
        print(f"  {name:<15} {_s(value)}")
    print(f"  ad - bc = {_s(det_2x2(M))}")

    house = [(0, 0), (4, 0), (4, 3), (2, 5), (0, 3)]  # counterclockwise pentagon
    for A in ([[3, 1], [1, 2]], [[1, 2], [3, 1]], [[2, 4], [1, 2]]):
        before, after = signed_area(house), signed_area(transform(A, house))
        print(f"\nA = {A}: det {_s(det(A))}")
        print(f"  house area {_s(before)} -> {_s(after)} (ratio {_s(after / before)})")

    B = [[0, 2, 1], [1, 1, 0], [2, 0, 3]]
    f = lu(B)
    pivots = [f.U[i][i] for i in range(3)]
    print(f"\nB = {B}")
    print(f"  row order after swaps {f.perm}, sign {permutation_sign(f.perm):+d}")
    print(f"  pivots {[_s(p) for p in pivots]} -> det = {_s(det(B))}")
