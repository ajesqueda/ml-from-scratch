"""Turn messy linear equations into the matrix form Ax = b.

    "y + 2x = 5"      ->   2x + 1y + 0z = 5
    "3x = 4 + z"      ->   3x + 0y - 1z = 4
    "z - y = 0"       ->   0x - 1y + 1z = 0

The recipe, the same one you'd follow by hand:

    1. fix the variable order (one column per variable)
    2. move every variable term to the left and every number to the right,
       flipping the sign of anything that crosses "="
    3. read off the coefficients: 1 for a bare variable, 0 for a missing one
    4. each equation is one row of A, and its right side is one entry of b

Run `python equations.py` to watch each equation get tidied.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

# A term is an optional coefficient (2, 0.5, 3/2) and an optional variable
# (x, y, x1, x_2), with an optional * between: "2x", "3/2 y", "4*z", "x", "7".
TERM = re.compile(
    r"^(?P<coef>\d+(?:\.\d+)?(?:/\d+)?)?\s*\*?\s*(?P<var>[A-Za-z][A-Za-z0-9_]*)?$"
)


@dataclass(frozen=True)
class Term:
    coef: Fraction
    var: str | None  # None means a plain number

    def __str__(self) -> str:
        sign = "-" if self.coef < 0 else "+"
        size = abs(self.coef)
        if self.var is None:
            return f"{sign} {size}"
        return f"{sign} {'' if size == 1 else size}{self.var}"


def format_terms(terms: Sequence[Term]) -> str:
    """Join terms the way you'd write them: no '+' in front of the first one."""
    if not terms:
        return "0"
    text = " ".join(map(str, terms))
    return text[2:] if text.startswith("+ ") else "-" + text[2:]


def parse_side(text: str) -> list[Term]:
    """Split one side of an equation into signed terms."""
    text = text.strip()
    if not text:
        raise ValueError("empty side of an equation")
    # Put a sign in front of every term so a leading term like "2x" becomes "+2x".
    if text[0] not in "+-":
        text = "+" + text
    # Split on signs, keeping them: "+2x-y" -> ["", "+", "2x", "-", "y"]. A sign
    # with nothing after it (like "2x + = 3") leaves an empty body and is rejected.
    pieces = re.split(r"([+-])", text)[1:]
    terms = []
    for sign, body in zip(pieces[::2], pieces[1::2], strict=True):
        match = TERM.match(body.strip())
        if not match or not (match["coef"] or match["var"]):
            raise ValueError(f"can't read term {sign}{body.strip()!r}")
        # A bare variable like "x" has an invisible coefficient of 1.
        coef = Fraction(match["coef"]) if match["coef"] else Fraction(1)
        terms.append(Term(-coef if sign == "-" else coef, match["var"]))
    return terms


@dataclass
class TidyEquation:
    """One equation after tidying: sum(coefs[v] * v) = constant."""

    source: str
    coefs: dict[str, Fraction]
    constant: Fraction
    steps: list[str]


def tidy(equation: str) -> TidyEquation:
    """Move variables left and numbers right, flipping signs that cross '='."""
    if equation.count("=") != 1:
        raise ValueError(f"expected exactly one '=' in {equation!r}")
    left_text, right_text = equation.split("=")
    left, right = parse_side(left_text), parse_side(right_text)

    moved_left = [t for t in left if t.var is not None]
    moved_right = [t for t in right if t.var is None]
    crossings = []
    for t in right:
        if t.var is not None:  # variable on the right -> moves left, sign flips
            flipped = Term(-t.coef, t.var)
            moved_left.append(flipped)
            crossings.append(f"{t} crosses '=' to the left as {flipped}")
    for t in left:
        if t.var is None:  # number on the left -> moves right, sign flips
            flipped = Term(-t.coef, None)
            moved_right.append(flipped)
            crossings.append(f"{t} crosses '=' to the right as {flipped}")

    coefs: dict[str, Fraction] = {}
    for t in moved_left:
        assert t.var is not None
        coefs[t.var] = coefs.get(t.var, Fraction(0)) + t.coef
    constant = sum((t.coef for t in moved_right), Fraction(0))

    steps = [f"start:     {equation.strip()}"]
    steps += [f"move:      {c}" for c in crossings] or ["move:      nothing crosses '='"]
    steps.append(f"regroup:   {format_terms(moved_left)} = {format_terms(moved_right)}")
    return TidyEquation(equation.strip(), coefs, constant, steps)


def _natural_key(name: str) -> tuple:
    # Sort x2 before x10, the way a person would.
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def _fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def parse_system(
    equations: Sequence[str], variables: Sequence[str] | None = None, explain: bool = False
) -> tuple[list[str], list[list[Fraction]], list[Fraction]]:
    """Return (variable order, A, b) for a list of equation strings.

    The variable order defaults to sorted (x, y, z or x1, x2, ..., x10), since
    the order the variables happen to appear in is not a reliable column order.
    """
    tidied = [tidy(eq) for eq in equations]
    found = {v for t in tidied for v in t.coefs}
    if variables is None:
        order = sorted(found, key=_natural_key)
    else:
        order = list(variables)
        missing = found - set(order)
        if missing:
            raise ValueError(f"variables not in the given order: {sorted(missing)}")

    # A variable missing from an equation still gets a column, holding 0.
    A = [[t.coefs.get(v, Fraction(0)) for v in order] for t in tidied]
    b = [t.constant for t in tidied]

    if explain:
        print(f"variable order: {', '.join(order)}\n")
        for i, (t, row, rhs) in enumerate(zip(tidied, A, b, strict=True), start=1):
            print(f"equation {i}")
            for step in t.steps:
                print("  " + step)
            # Every column written out, including 1s and 0s, so nothing is implicit.
            cells = [
                f"{'+' if c >= 0 else '-'} {_fmt(abs(c))}{v}"
                for c, v in zip(row, order, strict=True)
            ]
            filled = " ".join(cells)
            filled = filled[2:] if filled.startswith("+ ") else "-" + filled[2:]
            print(f"  fill 0s:   {filled} = {_fmt(rhs)}")
            print(f"  row {i}:     [{', '.join(_fmt(c) for c in row)}]   b{i} = {_fmt(rhs)}\n")
    return order, A, b


def main() -> None:
    from gauss import solve_system

    system = ["y + 2x = 5", "3x = 4 + z", "z - y = 0"]
    order, A, b = parse_system(system, explain=True)
    print("A =", [[_fmt(c) for c in row] for row in A])
    print("b =", [_fmt(v) for v in b])
    print("\n" + solve_system(A, b).describe() + f"   (order: {', '.join(order)})")


if __name__ == "__main__":
    main()
