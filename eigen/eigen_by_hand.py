"""2×2 eigenvalues and eigenvectors the way you'd find them by hand.

    eig2x2(A)              eigenvalues from trace and det: λ = m ± √(m² − p)
    eigenvector(A, lam)    solve (A − λI)x = 0 with the "swap and negate" rule
    eigenpairs(A)          every eigenvalue with its eigenvectors (handles repeats)
    power_iteration(A)     λ₁, the biggest |λ|, found by multiplying over and over

Exact with fractions whenever √(m² − p) is rational, so (1, −2) comes out as
(1, −2) and not (0.4472, −0.8944). Irrational eigenvalues fall back to floats.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import gcd, isqrt, lcm, sqrt

import numpy as np

Number = int | float | Fraction
TOL = 1e-12


class NoRealEigenvalues(ValueError):
    """m² − p < 0: the matrix turns every arrow, so nothing is only stretched."""


def _matrix(A: Sequence[Sequence[Number]]) -> tuple[Number, Number, Number, Number]:
    (a, b), (c, d) = A

    def exact(x: Number) -> Number:
        return x if isinstance(x, float) else Fraction(x)

    return exact(a), exact(b), exact(c), exact(d)


def _rational_sqrt(x: Fraction) -> Fraction | None:
    """√x as a fraction if it is one (x = 9/4 -> 3/2), else None."""
    if x < 0:
        return None
    n, d = isqrt(x.numerator), isqrt(x.denominator)
    return Fraction(n, d) if n * n == x.numerator and d * d == x.denominator else None


def trace_det(A) -> tuple[Number, Number]:
    a, b, c, d = _matrix(A)
    return a + d, a * d - b * c


def eig2x2(A) -> list[Number]:
    """Both eigenvalues, biggest |λ| first (λ₁ keeps its sign).

    det(λI − A) = λ² − (trace)λ + det = 0, so with m = trace/2 and p = det,
    λ = m ± √(m² − p). The two add to the trace and multiply to the det.
    """
    t, p = trace_det(A)
    m = t / 2
    disc = m * m - p
    if disc < 0:
        raise NoRealEigenvalues(f"m² − p = {disc} < 0: no real eigenvalues")
    root = _rational_sqrt(disc) if isinstance(disc, Fraction) else None
    if root is None:
        root = sqrt(disc)
        m = float(m)
    return sorted([m + root, m - root], key=lambda lam: -abs(lam))


def _is_zero(x: Number) -> bool:
    return x == 0 if isinstance(x, Fraction) else abs(x) < 1e-9


def _whole_numbers(v: list[Fraction]) -> list[int]:
    """Smallest whole numbers pointing the same way, first non-zero entry positive."""
    scale = lcm(*(x.denominator for x in v))
    ints = [int(x * scale) for x in v]
    g = gcd(*ints) or 1
    ints = [x // g for x in ints]
    first = next(x for x in ints if x != 0)
    return [-x for x in ints] if first < 0 else ints


def _tidy_float(v: list[float]) -> list[float]:
    """Unit length, first non-zero entry positive (eigenvectors only matter up to scale)."""
    arr = np.array(v, dtype=float)
    arr /= np.linalg.norm(arr)
    first = next(x for x in arr if abs(x) > TOL)
    return list(-arr if first < 0 else arr)


def eigenvector(A, lam: Number) -> list[list[Number]]:
    """Eigenvectors for λ: a list with one vector, or two when A = λI (every vector works).

    Subtract λ from the diagonal ONLY. The rows of A − λI must then be multiples of
    each other (det = 0), so there's one real equation a·x₁ + b·x₂ = 0. Instead of
    picking a free variable and dividing, take v = (−b, a): it satisfies
    a·(−b) + b·a = 0 with no division, so there's no wrong choice to make.
    """
    a, b, c, d = _matrix(A)
    if isinstance(lam, float):
        a, b, c, d = (float(x) for x in (a, b, c, d))
    else:
        lam = Fraction(lam)
    m11, m12, m21, m22 = a - lam, b, c, d - lam
    det = m11 * m22 - m12 * m21
    if not _is_zero(det):
        raise ValueError(f"{lam} isn't an eigenvalue: the rows of A − λI aren't multiples")
    if all(_is_zero(x) for x in (m11, m12, m21, m22)):
        return [[1, 0], [0, 1]]  # A − λI = 0 means A = λI: every vector is an eigenvector
    row = (m11, m12) if not (_is_zero(m11) and _is_zero(m12)) else (m21, m22)
    v = [-row[1], row[0]]
    if all(isinstance(x, Fraction) for x in v):
        return [_whole_numbers(v)]
    return [_tidy_float([float(x) for x in v])]


def eigenpairs(A) -> list[tuple[Number, list[list[Number]]]]:
    """Each distinct eigenvalue with its eigenvectors, biggest |λ| first.

    A repeated eigenvalue can have two independent eigenvectors (A = cI) or only
    one (a shear): the second case is "defective" and can't be diagonalized.
    """
    lams = eig2x2(A)
    distinct = [lams[0]] if lams[0] == lams[1] else lams
    return [(lam, eigenvector(A, lam)) for lam in distinct]


def power_iteration(A, steps: int = 200, start=(1.0, 0.3)) -> tuple[float, np.ndarray]:
    """Multiply a start vector by A over and over; it swings onto λ₁'s eigenvector.

    Each multiply scales the λ₁ part by |λ₁| and the λ₂ part by |λ₂|, so the λ₂ part
    fades by |λ₂/λ₁| per step. Rescale every step so nothing overflows. The Rayleigh
    quotient v·Av / v·v reads off λ₁ with its sign, even when λ₁ < 0 makes v flip
    direction on every step.
    """
    M = np.array(A, dtype=float)
    v = np.array(start, dtype=float)
    for _ in range(steps):
        v = M @ v
        v /= np.linalg.norm(v)
    lam = float(v @ M @ v / (v @ v))
    return lam, np.array(_tidy_float(list(v)))


def _s(x: Number) -> str:
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    return f"{x:.4f}"


if __name__ == "__main__":
    examples = {
        "two clean eigenvalues": [[4, 1], [2, 3]],
        "negative λ₁ (sign flip drill)": [[-4, 3], [2, 1]],
        "irrational eigenvalues": [[2, 1], [1, 1]],
        "repeated, only one eigenvector (shear)": [[3, 1], [0, 3]],
        "repeated, every vector (A = 3I)": [[3, 0], [0, 3]],
    }
    for name, A in examples.items():
        t, p = trace_det(A)
        print(f"{name}: A = {A}, trace {_s(t)}, det {_s(p)}")
        for lam, vecs in eigenpairs(A):
            shown = ", ".join("(" + ", ".join(_s(Fraction(x)) if isinstance(x, int) else _s(x)
                                               for x in v) + ")" for v in vecs)
            print(f"  λ = {_s(lam)}: {shown}")
    try:
        eig2x2([[0, -1], [1, 0]])
    except NoRealEigenvalues as e:
        print(f"90° rotation [[0, -1], [1, 0]]: {e}")

    for A in ([[4, 1], [2, 3]], [[-4, 3], [2, 1]]):
        lam, v = power_iteration(A)
        print(f"power iteration on {A}: λ₁ ≈ {lam:.6f}, v ≈ {np.round(v, 4)}")
