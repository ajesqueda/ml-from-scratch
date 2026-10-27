"""Gaussian elimination from scratch, with exact fractions and a step-by-step trace.

    forward pass   clear everything *below* each pivot  -> row echelon form (REF)
    backward pass  scale each pivot to 1, clear *above* -> reduced row echelon form (RREF)

Everything uses `fractions.Fraction`, so 3/2 stays 3/2 instead of becoming
1.4999999999. That keeps the trace readable and makes "is this entry zero?"
an exact question instead of a floating-point tolerance.

Run `python gauss.py` to watch a system get solved one row operation at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

Matrix = list[list[Fraction]]
Number = int | float | Fraction


def to_matrix(rows: Sequence[Sequence[Number]]) -> Matrix:
    """Copy into a list of Fraction rows. Floats are converted exactly as stored."""
    matrix = [[Fraction(x) for x in row] for row in rows]
    if len({len(row) for row in matrix}) > 1:
        raise ValueError("all rows must have the same length")
    return matrix


def _fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def format_rows(rows: Sequence[Sequence[Fraction]], labels: Sequence[str], bar: int | None) -> str:
    """Render rows with every column padded to the same width, so entries line up.

    `bar` draws a | before that column index, for augmented matrices [A | b].
    """
    cells = [[_fmt(x) for x in row] for row in rows]
    widths = [max(len(row[j]) for row in cells) for j in range(len(cells[0]))]
    label_width = max(len(label) for label in labels)
    lines = []
    for label, row in zip(labels, cells, strict=True):
        parts = [cell.rjust(widths[j]) for j, cell in enumerate(row)]
        if bar is not None:
            parts.insert(bar, "|")
        lines.append(f"  {label.rjust(label_width)}  [ {'  '.join(parts)} ]")
    return "\n".join(lines)


@dataclass
class Eliminator:
    """Runs row operations on a matrix and optionally prints each one."""

    matrix: Matrix
    trace: bool = False
    bar: int | None = None  # column where the augmented part starts, for display
    steps: list[str] = field(default_factory=list)

    def _log(self, text: str) -> None:
        self.steps.append(text.splitlines()[0])
        if self.trace:
            print(text + "\n")

    def swap(self, i: int, j: int) -> None:
        m = self.matrix
        m[i], m[j] = m[j], m[i]
        self._log(f"R{i + 1} <-> R{j + 1}   (pivot position held a 0)")

    def scale(self, i: int, factor: Fraction) -> None:
        if factor == 0:
            raise ValueError("scaling by 0 is not a legal row operation")
        before = self.matrix[i]
        self.matrix[i] = [factor * x for x in before]
        self._log(
            f"R{i + 1} <- ({_fmt(factor)})*R{i + 1}\n"
            + format_rows([before, self.matrix[i]], [f"R{i + 1}", "new"], self.bar)
        )

    def add_multiple(self, target: int, source: int, multiplier: Fraction) -> None:
        """R_target <- R_target - multiplier * R_source, one column at a time."""
        src, before = self.matrix[source], self.matrix[target]
        self.matrix[target] = [t - multiplier * s for t, s in zip(before, src, strict=True)]
        # Subtracting a negative multiple reads better as adding a positive one.
        sign, shown = ("-", multiplier) if multiplier > 0 else ("+", -multiplier)
        scaled = [shown * x for x in src]
        # The scaled source row is printed directly above the target row, so
        # each column is visibly "this entry, plus or minus the one above it".
        self._log(
            f"R{target + 1} <- R{target + 1} {sign} ({_fmt(shown)})*R{source + 1}\n"
            + format_rows(
                [src, scaled, before, self.matrix[target]],
                [f"R{source + 1}", f"{_fmt(shown)}*R{source + 1}", f"R{target + 1}", "new"],
                self.bar,
            )
        )

    def forward(self, n_cols: int | None = None) -> list[int]:
        """Forward pass to row echelon form. Returns the pivot column of each pivot row.

        `n_cols` limits which columns may hold pivots, so the right-hand side of
        [A | b] never gets one.
        """
        m = self.matrix
        n_cols = len(m[0]) if n_cols is None else n_cols
        pivots: list[int] = []
        row = 0
        for col in range(n_cols):
            if row == len(m):
                break
            # Use the first row at or below `row` with a non-zero entry here.
            # (Floating-point code swaps in the *largest* entry instead, called
            # partial pivoting, to limit rounding error. With exact fractions
            # any non-zero pivot is equally good.)
            candidate = next((r for r in range(row, len(m)) if m[r][col] != 0), None)
            if candidate is None:
                continue  # no pivot in this column: it will be a free variable
            if candidate != row:
                self.swap(row, candidate)
            for below in range(row + 1, len(m)):
                if m[below][col] != 0:
                    # multiplier = (number to clear) / (pivot)
                    self.add_multiple(below, row, m[below][col] / m[row][col])
            pivots.append(col)
            row += 1
        return pivots

    def backward(self, pivots: Sequence[int]) -> None:
        """Backward pass: pivots become 1 and everything above them becomes 0."""
        m = self.matrix
        for row in reversed(range(len(pivots))):
            col = pivots[row]
            if m[row][col] != 1:
                self.scale(row, 1 / m[row][col])
            for above in range(row):
                if m[above][col] != 0:
                    self.add_multiple(above, row, m[above][col])


def row_echelon(rows: Sequence[Sequence[Number]]) -> tuple[Matrix, list[int]]:
    e = Eliminator(to_matrix(rows))
    pivots = e.forward()
    return e.matrix, pivots


def rank(rows: Sequence[Sequence[Number]]) -> int:
    """Number of pivots, i.e. non-zero rows after the forward pass."""
    return len(row_echelon(rows)[1])


def rref(rows: Sequence[Sequence[Number]], trace: bool = False) -> tuple[Matrix, list[int]]:
    """Reduced row echelon form and its pivot columns. RREF is unique for a matrix."""
    e = Eliminator(to_matrix(rows), trace=trace)
    pivots = e.forward()
    e.backward(pivots)
    return e.matrix, pivots


@dataclass
class Solution:
    """Result of solving Ax = b.

    For kind == "infinite", every solution is
        x = particular + t1 * directions[0] + t2 * directions[1] + ...
    with one free parameter per free variable.
    """

    kind: Literal["none", "unique", "infinite"]
    particular: list[Fraction] | None = None
    directions: list[list[Fraction]] = field(default_factory=list)
    free_vars: list[int] = field(default_factory=list)
    bound_vars: list[int] = field(default_factory=list)

    def describe(self) -> str:
        if self.kind == "none":
            return "no solution (a row reads 0 = non-zero)"
        assert self.particular is not None
        base = "[" + ", ".join(_fmt(x) for x in self.particular) + "]"
        if self.kind == "unique":
            return f"unique solution x = {base}"
        terms = [
            f"t{k + 1}*[" + ", ".join(_fmt(x) for x in d) + "]"
            for k, d in enumerate(self.directions)
        ]
        free = ", ".join(f"x{v + 1}" for v in self.free_vars)
        return f"infinitely many: x = {base} + " + " + ".join(terms) + f"   (free: {free})"


def solve_system(
    A: Sequence[Sequence[Number]], b: Sequence[Number], trace: bool = False
) -> Solution:
    """Classify and solve Ax = b by row-reducing the augmented matrix [A | b]."""
    a = to_matrix(A)
    n = len(a[0])
    if len(b) != len(a):
        raise ValueError("b needs one entry per row of A")
    augmented = [row + [Fraction(v)] for row, v in zip(a, b, strict=True)]

    e = Eliminator(augmented, trace=trace, bar=n)
    pivots = e.forward(n_cols=n)
    m = e.matrix
    # Below the last pivot every coefficient is 0. A non-zero right-hand side
    # there is the equation 0 = c, which nothing can satisfy.
    if any(m[r][n] != 0 for r in range(len(pivots), len(m))):
        return Solution("none")
    e.backward(pivots)

    free = [c for c in range(n) if c not in pivots]
    # Particular solution: set every free variable to 0 and read off the
    # bound ones straight from the right-hand column.
    particular = [Fraction(0)] * n
    for row, col in enumerate(pivots):
        particular[col] = m[row][n]
    if not free:
        return Solution("unique", particular, bound_vars=list(pivots))

    # One direction per free variable: set it to 1, the other free ones to 0,
    # and each bound variable moves by minus its coefficient in that column.
    directions = []
    for f in free:
        d = [Fraction(0)] * n
        d[f] = Fraction(1)
        for row, col in enumerate(pivots):
            d[col] = -m[row][f]
        directions.append(d)
    return Solution("infinite", particular, directions, free, list(pivots))


def inverse(A: Sequence[Sequence[Number]], trace: bool = False) -> Matrix:
    """Invert A by row-reducing [A | I]. The right block ends up holding A^-1."""
    a = to_matrix(A)
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("only square matrices have inverses")
    identity = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    e = Eliminator([row + id_row for row, id_row in zip(a, identity, strict=True)], trace, bar=n)
    pivots = e.forward(n_cols=n)
    if len(pivots) < n:
        raise ValueError(f"matrix is singular (rank {len(pivots)} < {n})")
    e.backward(pivots)
    return [row[n:] for row in e.matrix]


def main() -> None:
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    print("Solve Ax = b, tracing every row operation on [A | b]:\n")
    print(solve_system(A, b, trace=True).describe())

    print("\nA dependent system (row 3 = row 1 + row 2):")
    print(solve_system([[1, 1, 2], [2, 3, 5], [3, 4, 7]], [3, 8, 11]).describe())
    print("\nAn inconsistent one (same left side, right side doesn't add up):")
    print(solve_system([[1, 1, 2], [2, 3, 5], [3, 4, 7]], [3, 8, 12]).describe())


if __name__ == "__main__":
    main()
