# Gaussian elimination, step by step

Row reduction from scratch with exact fractions. It solves `Ax = b`, says whether
there are **0, 1 or infinitely many** solutions, writes infinite solutions in
parametric form, and inverts matrices by reducing `[A | I]`. A trace mode prints
every row operation with the columns lined up.

It also turns messy written equations like `3x = 4 + z` into `A` and `b`, and shows
matrix × vector both ways (row view and column view).

> Concept from BU DX601 (Mathematical Foundations of ML), Week 9: systems of linear
> equations. All code is my own.

## Why it matters

This is what's underneath `np.linalg.solve` and `np.linalg.inv`, and it's how
least squares gets solved. Rank, free variables and "no solution" all come straight
out of the elimination. They decide whether a regression has a unique answer or
whether some of your features are just copies of others.

## The algorithm

```
forward pass   for each column: pick a non-zero pivot (swap rows if it's 0),
               clear every entry below it      -> row echelon form, rank = # pivots
backward pass  scale each pivot to 1, clear every entry above it
                                                -> reduced row echelon form (unique)
```

The multiplier is always **(number to clear) ÷ (pivot)**. Using
`fractions.Fraction` keeps it exact (3/2, not 1.4999…), which also makes "is this
zero?" an exact test.

## Run it

```bash
pip install -r requirements-dev.txt
python gaussian-elimination/gauss.py
pytest gaussian-elimination
```

Each step prints the source row, the scaled copy, then the target row, in that order
top to bottom, so every column is just "this entry ± the one above it":

```
R2 <- R2 + (3/2)*R1
      R1  [  2    1    -1  |    8 ]
  3/2*R1  [  3  3/2  -3/2  |   12 ]
      R2  [ -3   -1     2  |  -11 ]
     new  [  0  1/2   1/2  |    1 ]
...
unique solution x = [2, 3, -1]

A dependent system (row 3 = row 1 + row 2):
infinitely many: x = [1, 2, 0] + t1*[-1, -1, 1]   (free: x3)

An inconsistent one (same left side, right side doesn't add up):
no solution (a row reads 0 = non-zero)
```

## From messy equations to `Ax = b`

`equations.py` does the tidying step you'd otherwise do by hand, and `explain=True`
prints each move:

```bash
python gaussian-elimination/equations.py
```

```
equation 2
  start:     3x = 4 + z
  move:      + z crosses '=' to the left as - z
  regroup:   3x - z = 4
  fill 0s:   3x + 0y - 1z = 4
  row 2:     [3, 0, -1]   b2 = 4
...
A = [['2', '1', '0'], ['3', '0', '-1'], ['0', '-1', '1']]
b = ['5', '4', '0']

unique solution x = [9/5, 7/5, 7/5]   (order: x, y, z)
```

The rules it follows:

1. **Fix the variable order.** By default it's sorted (`x, y, z` or `x1, x2, …, x10`),
   not the order the variables happen to appear in.
2. **Variables go left, numbers go right.** Anything that crosses `=` flips its sign.
3. **Nothing is left implicit.** A bare `x` means `1x`, and a missing variable gets a
   `0` in its column.
4. **One equation is one row.** Its right side becomes that row's entry of `b`.

Flipping every sign in an equation, `b` included, gives the same system. The tests
check that both versions have the same solution.

## Matrix × vector, two views

`matvec.py` computes `Ax` two ways and the tests check they always agree with each
other and with NumPy:

```
row view:     [8, 9]   = [2*3 + 1*2, 1*3 + 3*2]     each entry = one row · x
column view:  [8, 9]   = 3*[2, 1] + 2*[1, 3]        x is a recipe for mixing columns
```

The row view is how you compute by hand. The column view is how to *think* about
`Ax = b`: which mix of A's columns produces `b`? An m × n matrix takes a vector with n
entries and gives back m.

## API

| Function | Returns |
|----------|---------|
| `rref(M, trace=False)` | `(reduced matrix, pivot columns)` |
| `row_echelon(M)`, `rank(M)` | Forward pass only; rank = number of pivots |
| `solve_system(A, b, trace=False)` | `Solution` with `kind` of `"none"`, `"unique"` or `"infinite"`, plus `particular`, `directions`, `free_vars`, `bound_vars` |
| `inverse(A, trace=False)` | A⁻¹ from `[A \| I]`, or `ValueError` if A is singular |
| `parse_system(equations, variables=None, explain=False)` | `(variable order, A, b)` from equation strings |
| `matvec_rows(A, x)`, `matvec_columns(A, x)` | `Ax` by the row view and the column view |

## Reading a parametric solution

Free variables are the columns without a pivot, and bound variables are the ones with
a pivot. Set every free variable to 0 to get one **particular** solution. Then each
free variable adds a **direction** you can move along without breaking any equation:

```
x = [1, 2, 0] + t·[-1, -1, 1]     t = 0 → (1, 2, 0),  t = 1 → (0, 1, 1),  ...
```

The tests plug random values of `t` back into `Ax` and check they still give `b`.

## When to swap rows

- **With exact fractions** (this code): swap only when the pivot position holds 0.
  Any non-zero pivot works.
- **With floats** (NumPy/LAPACK): always swap in the row with the *largest* entry in
  that column. This is called **partial pivoting**. Dividing by a tiny pivot such as
  1e-17 blows rounding error up into garbage, and the largest pivot keeps every
  multiplier at most 1.

## What the tests check (245)

- Rank matches `np.linalg.matrix_rank` on 40 random matrices, half of them built to
  have low rank.
- RREF matches `sympy.Matrix.rref` **exactly**, fraction for fraction, on 40 random
  matrices.
- Unique solutions match `np.linalg.solve`, and inverses match `np.linalg.inv`.
- For 25 random consistent systems with low rank:
  - the number of free variables is `n − rank`
  - every point on the parametric solution satisfies `Ax = b`
- No solution, 0 pivots, forced row swaps, wide systems and singular matrices are
  each covered.
- Every row operation can be undone. Random sequences of swaps, scalings and
  add-a-multiples are reversed by applying the inverse moves in **reverse order**, which
  restores the original matrix, and the rank never changes along the way. Undoing in
  the forward order does not.
- The equation parser handles sign flips across `=`, like terms, fractions and
  decimals, `x1…x10` ordering and missing variables, and it rejects malformed input.
- The row and column views of `Ax` match `A @ x` on 30 random shapes, and wrong shapes
  are rejected.
