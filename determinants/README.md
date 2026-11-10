# Determinants as area scale factors

A 2×2 matrix sends the unit square to a parallelogram, and the determinant says how
much area got multiplied by, and whether the shape got mirrored. This folder computes
determinants **exactly** from the LU factorization built in
[`gaussian-elimination/`](../gaussian-elimination/#triangular-solves-and-lu-why-libraries-factor-once),
proves the 2×2 area formula without a graph, and checks that every shape scales by
|det A|.

> Concept from BU DX601 (Math for Data Science), Week 11: matrix invertibility and determinants.

## What's inside (`det_area.py`)

| Function | What it does |
|----------|--------------|
| `det(A)` | Exact determinant from `P A = L U`: product of U's pivots, sign flipped once per row swap |
| `det_2x2(A)` | ad − bc |
| `box_proof(a, b, c, d)` | ad − bc as a bounding box minus its corner pieces |
| `signed_area(polygon)` | Shoelace formula: positive counterclockwise, negative clockwise |
| `transform(A, polygon)` | Apply A to every vertex |
| `permutation_sign(perm)` | +1 or −1 for an even or odd number of row swaps |

## Run it

```bash
pip install -r requirements-dev.txt
python determinants/det_area.py
pytest determinants
```

## Area without a graph

The columns of A are where the unit square's edges land: `A·(1, 0)` is column 1 and
`A·(0, 1)` is column 2. For `[[3, 1], [1, 2]]` that's the parallelogram with sides
(3, 1) and (1, 2). Put it in a (3 + 1) × (1 + 2) box and subtract what's around it:

```
unit square under [[3, 1], [1, 2]]
  box             12      (a + b)(c + d)
  triangles a,c   3       two triangles, each a·c/2
  triangles b,d   2       two triangles, each b·d/2
  rectangles b,c  2       two rectangles, each b·c
  area            5
  ad - bc = 5
```

`(a + b)(c + d) − ac − bd − 2bc` expands to exactly `ad − bc`. The picture needs
positive entries, but the algebra doesn't, and the tests check the identity on random
integers.

## Any shape, and what the sign means

Squares aren't special. Every shape's area gets multiplied by |det A|, because any
shape can be cut into tiny squares and each one scales the same way. Applied to a
pentagon:

```
A = [[3, 1], [1, 2]]: det 5     house area 16 -> 80  (ratio 5)
A = [[1, 2], [3, 1]]: det -5    house area 16 -> -80 (ratio -5)
A = [[2, 4], [1, 2]]: det 0     house area 16 -> 0   (ratio 0)
```

The shoelace formula gives a *signed* area: positive when the vertices go around
counterclockwise. A negative determinant reverses that order, which means the shape
came out mirrored. The area itself is the absolute value. A zero determinant flattens
everything onto a line, so no area survives and nothing can undo it: A has no inverse.

## The determinant from LU

Elimination already computes the determinant. `P A = L U`, where L has 1s on its
diagonal (so det L = 1), U is triangular (so det U is the product of its pivots), and
each row swap in P flips the sign:

```
B = [[0, 2, 1], [1, 1, 0], [2, 0, 3]]
  row order after swaps [1, 0, 2], sign -1
  pivots ['1', '2', '4'] -> det = -8
```

`np.linalg.det(B)` returns `-7.999999999999998`. It uses the same method in floating
point; with fractions the answer is exactly −8, and a singular matrix gives exactly 0
instead of something like 1e-17. That matters when the question is "is this matrix
invertible?"

The LU view also explains the row-operation rules, which the tests check:

| Row operation | Effect on det | Why |
|---------------|---------------|-----|
| Swap two rows | flips the sign | one more swap in P |
| Add a multiple of one row to another | none | that's an elimination step, recorded in L, whose det is 1 |
| Multiply one row by k | multiplies by k | one pivot gets k times bigger |

## Why the rows of A⁻¹ are "recipes"

`A⁻¹A = I` says: row i of A⁻¹ times column j of A is 1 when i = j and 0 otherwise. So
row i of A⁻¹ is a set of weights that cancels every column of A except column i.
Since `b = A x = x₁·(column 1) + … + xₙ·(column n)`, applying row i's weights to b
cancels every term except xᵢ:

```
(row i of A⁻¹) · b = x₁·0 + … + xᵢ·1 + … + xₙ·0 = xᵢ
```

That's why each row is a recipe that turns the right-hand sides into one unknown, for
any size, not just 2×2. The tests check it on random 2×2, 3×3 and 4×4 systems.

## Tests

`tests/test_det_area.py` (29 tests): `det` matches `np.linalg.det` on random 1×1 to
5×5 integer matrices and is exact where floating point isn't; it agrees with the 2×2
formula and with the diagonal product for triangular matrices; det = 0 exactly when
the rank drops and `inverse` refuses; the three row-operation rules; the product
rule `det(AB) = det(A)·det(B)` and `det(A⁻¹) = 1/det(A)`; the box proof on random
entries; the unit square landing on A's columns; signed area scaling by det for
random polygons; mirroring; and the inverse-row recipe property.
