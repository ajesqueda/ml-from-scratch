"""Subspaces as solution sets: null spaces, equations for a span, intersections.

Built on the exact-fraction elimination in ../gaussian-elimination/gauss.py, so
"is this entry zero?" is an exact question and bases come out as whole numbers.

    null_space(A)            basis of {x : Ax = 0}, one vector per free column
    equations_for_span(G)    rows a with a·g = 0 for every generator g (n − m of them)
    intersect(eqs_U, eqs_W)  stack both sets of equations, take the null space
    pivot_report(A)          the "pivot per row → rank → columns − rank" read-off
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from fractions import Fraction
from math import gcd, lcm
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gaussian-elimination"))
from gauss import Number, rank, rref  # noqa: E402


def as_integers(v: Sequence[Fraction]) -> list[int]:
    """Scale a vector to the smallest whole numbers pointing the same way.

    Any non-zero multiple of a null-space vector is still in the null space, so this
    is the "pick a value for the free unknown that avoids fractions" step.
    """
    v = [Fraction(x) for x in v]
    scale = lcm(*(x.denominator for x in v))
    ints = [int(x * scale) for x in v]
    common = gcd(*ints) or 1
    return [x // common for x in ints]


def null_space(A: Sequence[Sequence[Number]]) -> list[list[int]]:
    """Basis for the solutions of Ax = 0. Empty when only x = 0 solves it."""
    R, pivots = rref(A)
    n = len(R[0])
    basis = []
    for f in (c for c in range(n) if c not in pivots):
        v = [Fraction(0)] * n
        v[f] = Fraction(1)  # this free unknown = 1, the other free ones = 0
        for row, p in enumerate(pivots):
            v[p] = -R[row][f]  # each pivot unknown, solved in terms of the free one
        basis.append(as_integers(v))
    return basis


def equations_for_span(generators: Sequence[Sequence[Number]]) -> list[list[int]]:
    """Rows a with a·g = 0 for every generator g, so span(G) = {x : Ax = 0}.

    Each row is a solution of the homogeneous system "generators as rows", which
    makes this the null space again. An m-dim span in ℝⁿ gets n − m equations.
    """
    return null_space(generators)


def contains(equations: Sequence[Sequence[Number]], x: Sequence[Number]) -> bool:
    """Is x in the subspace these equations cut out? (No equations = all of ℝⁿ.)"""
    xs = [Fraction(v) for v in x]
    return all(sum(Fraction(a) * v for a, v in zip(row, xs, strict=True)) == 0
               for row in equations)


def intersect(eqs_U: Sequence[Sequence[Number]], eqs_W: Sequence[Sequence[Number]],
              n: int | None = None) -> list[list[int]]:
    """Basis of U ∩ W: a point is in both iff it satisfies both sets of equations."""
    stacked = [*eqs_U, *eqs_W]
    if not stacked:
        if n is None:
            raise ValueError("pass n when neither subspace has equations")
        return [[int(i == j) for j in range(n)] for i in range(n)]
    return null_space(stacked)


def dim_sum(gens_U: Sequence[Sequence[Number]], gens_W: Sequence[Sequence[Number]]) -> int:
    """dim(U + W): the rank of both generator sets together."""
    return rank([*gens_U, *gens_W])


def pivot_report(A: Sequence[Sequence[Number]]) -> str:
    """Spell out the read-off: pivot per row → rank → free columns → dimension."""
    R, pivots = rref(A)
    m, n = len(R), len(R[0])
    lines = []
    for r in range(m):
        if r < len(pivots):
            lines.append(f"  row {r + 1}: first non-zero in column x{pivots[r] + 1} -> pivot")
        else:
            lines.append(f"  row {r + 1}: all zeros -> a spare equation")
    free = [f"x{c + 1}" for c in range(n) if c not in pivots]
    lines.append(f"  rank = {len(pivots)} pivots; free columns: {', '.join(free) or 'none'}")
    lines.append(f"  solution space dimension = columns - rank = {n} - {len(pivots)} "
                 f"= {n - len(pivots)}")
    return "\n".join(lines)


def component_variances(X, center: bool = True) -> np.ndarray:
    """Variance along each principal direction (largest first), via the SVD.

    With center=False the "variances" are measured around the origin instead of
    around the data's mean, which is what goes wrong if PCA skips the subtraction.
    """
    X = np.asarray(X, dtype=float)
    if center:
        X = X - X.mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    return s**2 / (len(X) - 1)


def _fmt(vs: list[list[int]]) -> str:
    return ", ".join("(" + ", ".join(map(str, v)) + ")" for v in vs) or "only 0"


if __name__ == "__main__":
    A = [[1, 2, 0, 3, 1], [2, 4, 1, 7, 1], [1, 2, 1, 4, 0]]
    print("A =", A)
    print(pivot_report(A))
    print("null space basis:", _fmt(null_space(A)))

    G = [[3, 1, -2], [1, 4, 5]]
    eqs = equations_for_span(G)
    print("\nspan of", G, "is cut out by", eqs, "  np.cross:", np.cross(*G).tolist())

    plane = [[1, 0, 2, 0], [0, 1, 1, 1]]
    hyper = [[1, 1, 1, 1]]  # x1 + x2 + x3 + x4 = 0
    both = intersect(equations_for_span(plane), hyper)
    print(f"\nplane ∩ hyperplane in R^4: {_fmt(both)}  (dimension {len(both)})")

    rng = np.random.default_rng(10)
    basis = np.array(plane, dtype=float)
    X = rng.normal(size=(200, 2)) @ basis  # 200 points on a 2-D plane through 0 in R^4
    noisy = X + rng.normal(scale=0.01, size=X.shape)
    shifted = X + np.array([5.0, -3.0, 2.0, 4.0])  # same plane, moved off the origin
    np.set_printoptions(precision=4, suppress=True)
    print("\ncomponent variances, data on a 2-D plane in R^4 (expect 4 - 2 = 2 zeros)")
    print("  exact                ", component_variances(X))
    print("  noisy                ", component_variances(noisy))
    print("  shifted, centered    ", component_variances(shifted))
    print("  shifted, NOT centered", component_variances(shifted, center=False))
