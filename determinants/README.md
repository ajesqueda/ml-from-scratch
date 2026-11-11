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

## What's inside (`cofactor_cramer.py`)

| Function | What it does |
|----------|--------------|
| `det_cofactor(A, counter)` | Cofactor expansion along the row or column with the most zeros, signs + − + |
| `det_lu(A, counter)` | The elimination route, with its operations counted |
| `cramer(A, b, counter)` | xᵢ = det(Aᵢ) / det A, exact; refuses a singular A |
| `parallelogram_area(corners)` | Area from 4 corners in order; checks the opposite corner matches |
| `parallelepiped_volume(o, a, b, c)` | 3-D volume from three edges at one corner |
| `operation_counts(n)` | Multiplications + divisions for each method on a dense n×n matrix |

## Run it

```bash
pip install -r requirements-dev.txt
python determinants/det_area.py
python determinants/cofactor_cramer.py
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

## Three ways to a determinant, and what each one costs

By hand, a 3×3 or a matrix full of zeros is quickest with **cofactor expansion**:
expand along the row or column with the most zeros, since a 0 entry wipes out its
whole term, and use the checkerboard signs `+ − +`. `det_cofactor` picks that line the
same way:

```
A = [[2, 0, 1, 3], [1, 0, 0, 2], [4, 5, 1, 0], [0, 0, 3, 1]]
  cofactor (down column 2, three zeros): 20
  elimination:                          20
```

For anything bigger, count the work. Every function takes a counter, so this is
measured rather than timed:

```
multiplications + divisions for one dense n x n problem
 n   cofactor   elimination   cramer
 2          2             4       14
 3          9            13       55
 4         40            29      149
 5        205            54      329
 6      1,236            90      636
 7      8,659           139    1,119
 8     69,280           203    1,835
```

- **Cofactor expansion** grows like n!: an n×n expands into n determinants of size
  n − 1. At 20×20 that's more than 10¹⁸ operations.
- **Elimination** grows like n³/3. It's slower than cofactors only at n = 2 and 3,
  which is why both are fine by hand.
- **Cramer's rule** solves `Ax = b` with n + 1 determinants, so even with elimination
  inside it's about n⁴. One `lu` followed by `lu_solve` does the same job in n³.
  Cramer is still handy for a 2×2, or when you only need one unknown: that's two
  determinants, not n + 1.

## More rules, and det vs area

The rules all come from "det is the volume scale factor":

| Rule | Why |
|------|-----|
| det(AB) = det A · det B | doing B then A scales volume by both factors |
| det(A⁻¹) = 1 / det A | undoing A has to undo its scaling |
| det(Aᵀ) = det A | rows and columns play the same role in the formula |
| det(cA) = cⁿ det A | an n×n matrix has n rows, and each one gets scaled by c |

Combining two of them: for a 3×3 with det A = d, `det(c·A⁻¹) = c³ / d`. The tests
check that one on a real matrix.

Whether to keep the sign depends on what the question asks:

| The question asks for | Answer with |
|-----------------------|-------------|
| "the determinant" | det, sign included |
| "the area" or "the volume" | \|det\| |
| "is it invertible?" / "for which t is it singular?" | whether det = 0 (set det = 0 and solve for t) |

For a parallelogram given by its corners, subtract one corner from its two neighbours
to get the edges, put them in the columns, and take |det|. `parallelogram_area`
checks the 4th corner is actually where those edges meet, so corners in the wrong
order fail loudly instead of giving a wrong area.

## One fact, four faces

For a square matrix A, these all say the same thing, and the tests check all four
agree on random invertible and singular 4×4 matrices:

- A has no inverse
- rank A < n
- det A = 0
- Ax = 0 has a non-zero solution (the null space from
  [`vector-spaces/`](../vector-spaces/) isn't empty)

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

`tests/test_cofactor_cramer.py` (29 tests): cofactor expansion matches elimination
and `np.linalg.det` up to 5×5 and does less work along zeros; det(Aᵀ), det(cA) = cⁿ det A
and det(cA⁻¹); the four faces of singularity agree; Cramer's rule matches `lu_solve`
with either determinant method and refuses singular A; parallelogram areas from corners
(any direction, any position, wrong order rejected) and parallelepiped volumes; and the
operation counts grow factorially for cofactors but stay within n³ for elimination.

`tests/test_det_area.py` (29 tests): `det` matches `np.linalg.det` on random 1×1 to
5×5 integer matrices and is exact where floating point isn't; it agrees with the 2×2
formula and with the diagonal product for triangular matrices; det = 0 exactly when
the rank drops and `inverse` refuses; the three row-operation rules; the product
rule `det(AB) = det(A)·det(B)` and `det(A⁻¹) = 1/det(A)`; the box proof on random
entries; the unit square landing on A's columns; signed area scaling by det for
random polygons; mirroring; and the inverse-row recipe property.
