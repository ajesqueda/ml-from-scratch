from fractions import Fraction

import numpy as np
import pytest
from equations import parse_side, parse_system, tidy
from gauss import solve_system, to_matrix


def test_example_from_notes():
    order, A, b = parse_system(["y + 2x = 5", "3x = 4 + z", "z - y = 0"])
    assert order == ["x", "y", "z"]
    assert A == to_matrix([[2, 1, 0], [3, 0, -1], [0, -1, 1]])
    assert b == [5, 4, 0]


@pytest.mark.parametrize(
    "text,coefs,constant",
    [
        ("2x + 3y = 7", {"x": 2, "y": 3}, 7),
        ("x = y", {"x": 1, "y": -1}, 0),  # y crosses '=' and flips sign
        ("5 = x", {"x": -1}, -5),  # both sides cross
        ("x + 4 = 10", {"x": 1}, 6),  # 4 crosses to the right as -4
        ("-x - y = -3", {"x": -1, "y": -1}, -3),
        ("2x + x = 9", {"x": 3}, 9),  # like terms combine
        ("x - x + y = 1", {"x": 0, "y": 1}, 1),
        ("3/2 x + 0.5y = 1/4", {"x": Fraction(3, 2), "y": Fraction(1, 2)}, Fraction(1, 4)),
        ("4*z = 8", {"z": 4}, 8),
        ("x1 + x2 = 1 - x3", {"x1": 1, "x2": 1, "x3": 1}, 1),
    ],
)
def test_tidy(text, coefs, constant):
    t = tidy(text)
    assert t.coefs == coefs
    assert t.constant == constant


def test_missing_variables_get_zero_columns():
    _, A, _ = parse_system(["x = 1", "y = 2", "z = 3"])
    assert A == to_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_flipping_every_sign_gives_an_equivalent_system():
    _, A1, b1 = parse_system(["y + 2x = 5", "3x = 4 + z", "z - y = 0"])
    _, A2, b2 = parse_system(["-y - 2x = -5", "-3x = -4 - z", "y - z = 0"])
    assert solve_system(A1, b1).particular == solve_system(A2, b2).particular


def test_natural_variable_order_and_explicit_order():
    order, _, _ = parse_system(["x10 + x2 + x1 = 0"])
    assert order == ["x1", "x2", "x10"]
    order, A, _ = parse_system(["a + 2b = 3"], variables=["b", "a", "c"])
    assert order == ["b", "a", "c"]
    assert A == to_matrix([[2, 1, 0]])
    with pytest.raises(ValueError, match="not in the given order"):
        parse_system(["a + q = 1"], variables=["a"])


def test_parsed_system_reproduces_left_sides():
    # The check from the notes: A @ x gives back each equation's left side.
    system = ["y + 2x = 5", "3x = 4 + z", "z - y = 0"]
    _, A, b = parse_system(system)
    x = solve_system(A, b).particular
    assert x is not None
    np.testing.assert_allclose(
        np.array(A, dtype=float) @ np.array(x, dtype=float), np.array(b, dtype=float)
    )


def test_homogeneous_system_always_has_the_zero_solution():
    _, A, b = parse_system(["x + y + z = 0", "2x - y = 0", "3x + z = 0"])
    sol = solve_system(A, b)
    assert sol.kind in {"unique", "infinite"}
    if sol.kind == "unique":
        assert sol.particular == [0, 0, 0]


def test_inconsistent_parallel_lines():
    # Same slope, different intercepts: subtracting gives 0 = 2.
    _, A, b = parse_system(["x + y = 1", "x + y = 3"])
    assert solve_system(A, b).kind == "none"


@pytest.mark.parametrize("bad", ["x + y", "x = = 1", "2x + = 3", "x + ? = 1", " = 4"])
def test_malformed_equations_are_rejected(bad):
    with pytest.raises(ValueError):
        tidy(bad)


def test_parse_side_reads_implicit_coefficients():
    terms = parse_side("x - y + 7")
    assert [(t.coef, t.var) for t in terms] == [(1, "x"), (-1, "y"), (7, None)]


def test_explain_shows_each_crossing(capsys):
    parse_system(["3x = 4 + z"], explain=True)
    out = capsys.readouterr().out
    assert "+ z crosses '=' to the left as - z" in out
    assert "fill 0s:   3x - 1z = 4" in out
