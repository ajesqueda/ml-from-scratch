"""Triangular solves, LU factorization, a rank test and quadratic minima.

    forward_sub    solve L y = b, top to bottom      (L lower triangular)
    back_sub       solve U x = y, bottom to top      (U upper triangular)
    lu             factor P A = L U                  (elimination, with its multipliers kept)
    lu_solve       solve A x = b as forward_sub + back_sub on the saved factors

Why this matters: Gaussian elimination costs about n³/3 multiplications, but a
triangular solve costs about n²/2. LU pays the n³ part once, so each new
right-hand side b after that is just two n² triangular solves. That's how
numerical libraries solve the same system for many b's.

Everything uses exact fractions, like gauss.py. Each function can count its
multiplications and divisions in an optional `counter`, so the n² vs n³
difference can be measured exactly instead of timed.

Run `python triangular.py` to see the operation counts grow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from gauss import Matrix, Number, rank, to_matrix


class Counter:
    """Counts multiplications and divisions (the expensive operations)."""

    def __init__(self) -> None:
        self.ops = 0

    def tick(self, n: int = 1) -> None:
        self.ops += n


def _vector(b: Sequence[Number]) -> list[Fraction]:
    return [Fraction(v) for v in b]


def _check_square(M: Matrix, b: Sequence[Fraction]) -> int:
    n = len(M)
    if any(len(row) != n for row in M):
        raise ValueError("matrix must be square")
    if len(b) != n:
        raise ValueError(f"b has {len(b)} entries but the matrix has {n} rows")
    return n


def forward_sub(
    L: Sequence[Sequence[Number]], b: Sequence[Number], counter: Counter | None = None
) -> list[Fraction]:
    """Solve L y = b for lower-triangular L, first unknown first.

    Row i reads  L[i][0]*y0 + ... + L[i][i]*yi = b[i], and y0..y(i-1) are
    already known, so yi = (b[i] - known part) / L[i][i].
    """
    L, b = to_matrix(L), _vector(b)
    n = _check_square(L, b)
    if any(L[i][j] != 0 for i in range(n) for j in range(i + 1, n)):
        raise ValueError("forward_sub needs a lower-triangular matrix (zeros above the diagonal)")
    y: list[Fraction] = []
    for i in range(n):
        if L[i][i] == 0:
            raise ZeroDivisionError(
                f"L[{i}][{i}] is 0, so y{i} can't be pinned down: "
                "the system has no solution or infinitely many"
            )
        known = sum((L[i][j] * y[j] for j in range(i)), Fraction(0))
        y.append((b[i] - known) / L[i][i])
        if counter:
            counter.tick(i + 1)  # i multiplications + 1 division
    return y


def back_sub(
    U: Sequence[Sequence[Number]], b: Sequence[Number], counter: Counter | None = None
) -> list[Fraction]:
    """Solve U x = b for upper-triangular U, last unknown first."""
    U, b = to_matrix(U), _vector(b)
    n = _check_square(U, b)
    if any(U[i][j] != 0 for i in range(n) for j in range(i)):
        raise ValueError("back_sub needs an upper-triangular matrix (zeros below the diagonal)")
    x = [Fraction(0)] * n
    for i in reversed(range(n)):
        if U[i][i] == 0:
            raise ZeroDivisionError(
                f"U[{i}][{i}] is 0, so x{i} can't be pinned down: "
                "the system has no solution or infinitely many"
            )
        known = sum((U[i][j] * x[j] for j in range(i + 1, n)), Fraction(0))
        x[i] = (b[i] - known) / U[i][i]
        if counter:
            counter.tick(n - i)
    return x


@dataclass
class LU:
    """P A = L U, with P stored as `perm`: row i of PA is row perm[i] of A."""

    perm: list[int]
    L: Matrix
    U: Matrix


def lu(A: Sequence[Sequence[Number]], counter: Counter | None = None) -> LU:
    """Factor A by Gaussian elimination, keeping every multiplier in L.

    The forward pass of elimination turns A into U. Each step
    R_i <- R_i - m * R_k stores m at L[i][k], so L records exactly how to
    rebuild A from U. Rows are swapped only when a pivot is 0 (exact fractions
    don't need the "largest pivot" rule that floating point does).
    """
    U = to_matrix(A)
    n = len(U)
    if any(len(row) != n for row in U):
        raise ValueError("LU here is for square matrices")
    L = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    perm = list(range(n))

    for k in range(n):
        pivot_row = next((r for r in range(k, n) if U[r][k] != 0), None)
        if pivot_row is None:
            raise ZeroDivisionError(f"matrix is singular: no pivot in column {k}")
        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            perm[k], perm[pivot_row] = perm[pivot_row], perm[k]
            # Multipliers already found belong to the rows, so they move too.
            L[k][:k], L[pivot_row][:k] = L[pivot_row][:k], L[k][:k]
        for i in range(k + 1, n):
            m = U[i][k] / U[k][k]  # (number to clear) / (pivot)
            L[i][k] = m
            U[i] = [u_i - m * u_k for u_i, u_k in zip(U[i], U[k], strict=True)]
            if counter:
                counter.tick(1 + (n - k))  # one division + one multiply per column
    return LU(perm, L, U)


def lu_solve(factors: LU, b: Sequence[Number], counter: Counter | None = None) -> list[Fraction]:
    """Solve A x = b with saved factors: forward_sub then back_sub, about n² work."""
    b = _vector(b)
    permuted = [b[i] for i in factors.perm]  # apply the same row swaps to b
    y = forward_sub(factors.L, permuted, counter)
    return back_sub(factors.U, y, counter)


def classify_by_rank(
    A: Sequence[Sequence[Number]], b: Sequence[Number]
) -> Literal["none", "unique", "infinite"]:
    """0, 1 or infinitely many solutions from two ranks, without solving anything.

    If adding b raises the rank, some row reduces to 0 = non-zero: no solution.
    Otherwise there's one solution when every unknown gets a pivot, and
    infinitely many when some column has none (a free variable).
    """
    a = to_matrix(A)
    augmented = [row + [Fraction(v)] for row, v in zip(a, b, strict=True)]
    r_a, r_ab = rank(a), rank(augmented)
    if r_a < r_ab:
        return "none"
    return "unique" if r_a == len(a[0]) else "infinite"


def is_positive_definite(H: Sequence[Sequence[Number]]) -> bool:
    """True if H is symmetric and elimination finds only positive pivots, no swaps.

    For a symmetric matrix that's equivalent to xᵀHx > 0 for every x ≠ 0, which
    is what makes a quadratic bowl-shaped, with exactly one minimum.
    """
    h = to_matrix(H)
    n = len(h)
    if any(h[i][j] != h[j][i] for i in range(n) for j in range(n)):
        return False
    try:
        factors = lu(h)
    except ZeroDivisionError:
        return False
    return factors.perm == list(range(n)) and all(factors.U[i][i] > 0 for i in range(n))


def quadratic_minimum(H: Sequence[Sequence[Number]], g: Sequence[Number]) -> list[Fraction]:
    """Minimize f(x) = ½ xᵀ H x + gᵀ x by solving the linear system H x = -g.

    The gradient of f is H x + g. Setting it to 0 is a linear system, the same
    one you get by setting every partial derivative to 0 by hand.
    Example: f = x² + xy + y² - 3x - 3y has H = [[2, 1], [1, 2]], g = [-3, -3].
    """
    if not is_positive_definite(H):
        raise ValueError(
            "H must be symmetric positive definite; otherwise f has a saddle "
            "or no lowest point, and gradient = 0 isn't a minimum"
        )
    return lu_solve(lu(H), [-Fraction(v) for v in g])


def _demo_matrix(n: int) -> list[list[int]]:
    # Diagonally dominant, so no row swaps are needed and every size is solvable.
    return [[n + 1 if i == j else (i + 2 * j) % 3 for j in range(n)] for i in range(n)]


def main() -> None:
    print("Multiplications + divisions needed (exact counts):\n")
    print(f"{'n':>4}  {'LU factor':>10}  {'one solve with LU':>18}")
    previous = None
    for n in (10, 20, 40, 80):
        A, b = _demo_matrix(n), [1] * n
        factor_ops, solve_ops = Counter(), Counter()
        factors = lu(A, factor_ops)
        lu_solve(factors, b, solve_ops)
        growth = ""
        if previous:
            growth = (f"   x{factor_ops.ops / previous[0]:.1f} and "
                      f"x{solve_ops.ops / previous[1]:.1f} when n doubles")
        print(f"{n:>4}  {factor_ops.ops:>10,}  {solve_ops.ops:>18,}{growth}")
        previous = (factor_ops.ops, solve_ops.ops)

    print("\nMinimum of f = x² + xy + y² - 3x - 3y:")
    print("  x, y =", [str(v) for v in quadratic_minimum([[2, 1], [1, 2]], [-3, -3])])


if __name__ == "__main__":
    main()
