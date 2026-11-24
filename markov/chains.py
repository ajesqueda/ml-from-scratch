"""Markov chains as matrix powers, and the eigen-pieces trick behind them.

    steady_state(P)          the distribution s with P s = s and sum(s) = 1, exact
    distribution_after(P, s0, k)    where the chain is after k steps: P^k s0
    power_via_eigen(A, y, k) A^k y from eigen-pieces: c1 λ1^k v1 + c2 λ2^k v2
    mixing_factor(P)         |λ2|: how fast the chain forgets its starting state (float)

Columns are "from" and rows are "to", so every column of P sums to 1 and a
distribution is a column vector: next = P @ current. Exact fractions throughout
(floats are read as the decimal they print as, so 0.3 is exactly 3/10), except
mixing_factor, which is a floating-point eigenvalue estimate. Built on
../vector-spaces/subspaces.py (null spaces) and ../eigen/eigen_by_hand.py.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _folder in ("vector-spaces", "eigen"):
    sys.path.insert(0, str(_ROOT / _folder))
from eigen_by_hand import eigenpairs  # noqa: E402
from subspaces import null_space  # noqa: E402

Number = int | float | Fraction
Matrix = list[list[Fraction]]


def _frac(x: Number | str) -> Fraction:
    """A float becomes the decimal it prints as (0.3 -> 3/10, not 5404319552844595/2^54)."""
    return Fraction(str(x)) if isinstance(x, float) else Fraction(x)


def as_matrix(P: Sequence[Sequence[Number]]) -> Matrix:
    """Fractions from ints, Fractions, floats or decimal strings ("0.3" is exactly 3/10)."""
    return [[_frac(x) for x in row] for row in P]


def check_stochastic(P: Sequence[Sequence[Number]]) -> Matrix:
    """Columns are "from": each must be non-negative and sum to exactly 1."""
    M = as_matrix(P)
    n = len(M)
    if n == 0:
        raise ValueError("a transition matrix needs at least one state")
    if any(len(row) != n for row in M):
        raise ValueError("a transition matrix is square")
    for j in range(n):
        column = [M[i][j] for i in range(n)]
        if any(x < 0 for x in column):
            raise ValueError(f"column {j} has a negative probability")
        if sum(column) != 1:
            raise ValueError(f"column {j} sums to {sum(column)}, not 1 (columns are 'from')")
    return M


def swap_and_flip(a: Number, b: Number) -> list[Number]:
    """A non-zero solution of a·s1 + b·s2 = 0: swap the numbers, flip one sign."""
    if a == 0 and b == 0:
        raise ValueError("0·s1 + 0·s2 = 0 holds for every s: there's no equation to solve")
    return [b, -a]


def normalize(v: Sequence[Number]) -> list[Fraction]:
    """Divide EACH entry by the SUM of all entries, keeping the order."""
    v = [_frac(x) for x in v]
    total = sum(v)
    if total == 0:
        raise ValueError("entries sum to 0, so they can't be scaled to probabilities")
    return [x / total for x in v]


def steady_state(P: Sequence[Sequence[Number]]) -> list[Fraction]:
    """The eigenvector of P for λ = 1, scaled so its entries add to 1.

    Solve (P − I)s = 0. For a 2×2 the rows are multiples of each other, so there's
    one equation and swap-and-flip answers it. Bigger chains use the exact null
    space. A chain that splits into separate pieces has more than one steady
    state, so the question has no single answer and this raises.
    """
    M = check_stochastic(P)
    n = len(M)
    shifted = [[M[i][j] - (i == j) for j in range(n)] for i in range(n)]
    if n == 2:
        row = shifted[0] if any(shifted[0]) else shifted[1]
        if not any(row):  # P = I: every distribution stays put
            raise ValueError("P is the identity: every distribution is steady")
        return normalize(swap_and_flip(*row))
    basis = null_space(shifted)
    if len(basis) != 1:
        raise ValueError(f"{len(basis)} independent steady states: the chain splits into pieces")
    return normalize(basis[0])


def distribution_after(P, s0: Sequence[Number], k: int) -> list[Fraction]:
    """P^k s0 by multiplying k times (the "run k steps" question)."""
    M = check_stochastic(P)
    if k < 0:
        raise ValueError("k counts steps, so it can't be negative")
    if len(s0) != len(M):
        raise ValueError(f"s0 has {len(s0)} entries but the chain has {len(M)} states")
    s = [_frac(x) for x in s0]
    for _ in range(k):
        s = [sum(M[i][j] * s[j] for j in range(len(s))) for i in range(len(M))]
    return s


def mixing_factor(P) -> float:
    """|λ2| for the chain: the rate at which it forgets "where it started" (a float).

    The steady state is the λ = 1 piece, which never shrinks. Every other piece
    shrinks by its |λ| per step, so the slowest one, |λ2|, sets how long the chain
    remembers its start. |λ2| = 1 means it never forgets (it cycles or splits).

    For a 2-state chain the gap shrinks by exactly |λ2| every step. In bigger chains
    it's the long-run rate: a repeated λ without enough eigenvectors adds a factor
    like k·|λ2|^k, so early steps can shrink less. A one-state chain is always at
    its steady state, so there's nothing to forget: 0.
    """
    M = check_stochastic(P)
    if len(M) == 1:
        return 0.0
    lams = sorted(np.abs(np.linalg.eigvals(np.array(M, dtype=float))))
    return float(lams[-2])


def power_via_eigen(A, y: Sequence[Number], k: int) -> list[Fraction]:
    """A^k y without multiplying by A k times (2×2 with two distinct eigenvalues).

    Split y = c1·v1 + c2·v2. These are two DIFFERENT equations (one per entry), so
    solve one for c1 and substitute into the OTHER. Then every multiply by A just
    scales each piece by its own λ, so A^k y = c1·λ1^k·v1 + c2·λ2^k·v2.
    """
    if k < 0:
        raise ValueError("k counts steps, so it can't be negative")
    if len(y) != 2:
        raise ValueError("this version is for 2×2 matrices, so y has two entries")
    pairs = eigenpairs(as_matrix(A))  # floats read as decimals keep rational λ exact
    if len(pairs) != 2:
        raise ValueError("needs two distinct eigenvalues")
    (l1, (v1,)), (l2, (v2,)) = pairs
    if not all(isinstance(x, Fraction) for x in (l1, l2)):
        raise ValueError("this exact version needs rational eigenvalues")
    y1, y2 = (_frac(x) for x in y)
    # y1 = c1·v1[0] + c2·v2[0] and y2 = c1·v1[1] + c2·v2[1]: eliminate c1.
    det = v1[0] * v2[1] - v1[1] * v2[0]
    c2 = (v1[0] * y2 - v1[1] * y1) / det
    c1 = (y1 - c2 * v2[0]) / v1[0] if v1[0] != 0 else (y2 - c2 * v2[1]) / v1[1]
    return [c1 * l1**k * v1[i] + c2 * l2**k * v2[i] for i in range(2)]


def _s(v) -> str:
    return "(" + ", ".join(str(x) for x in v) + ")"


if __name__ == "__main__":
    P = [["0.9", "0.5"], ["0.1", "0.5"]]  # 2 weather states: sunny, rainy (columns = from)
    P = [[Fraction(x) for x in row] for row in P]
    s = steady_state(P)
    print("P =", [[str(x) for x in row] for row in P])
    print("  P − I row 1:", [str(x) for x in (P[0][0] - 1, P[0][1])],
          "-> swap and flip:", [str(x) for x in swap_and_flip(P[0][0] - 1, P[0][1])])
    print("  steady state:", _s(s), " check P s = s:", distribution_after(P, s, 1) == s)
    print(f"  mixing factor |λ2| = {mixing_factor(P):.2f}")
    for k in (1, 2, 5, 10, 20):
        d = distribution_after(P, [0, 1], k)
        print(f"  start rainy, after {k:2d} days: {[round(float(x), 4) for x in d]}")

    A = [[4, 1], [2, 3]]
    y = [7, 1]
    print("\nA^k y via eigen-pieces for A =", A, "y =", y)
    for k in (1, 3, 6):
        print(f"  k={k}: {_s(power_via_eigen(A, y, k))}  direct: "
              f"{(np.linalg.matrix_power(np.array(A), k) @ np.array(y)).tolist()}")
