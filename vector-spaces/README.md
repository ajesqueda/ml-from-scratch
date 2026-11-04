# Vector spaces: subspaces, orthonormal bases and projections

Two ways to describe a subspace, and the tools that move between them:

- **As solutions of equations** (`subspaces.py`): null spaces, the equations that cut
  out a span, intersections, and the "pivots → rank → columns − rank" read-off. Built on
  the exact-fraction elimination from [`gaussian-elimination/`](../gaussian-elimination/),
  so bases come out as whole numbers.
- **With an orthonormal basis** (`orthonormal.py`): orthonormal bases turn "solve a
  system" into "take dot products", and orthogonal projection is the closest-point
  operation behind least squares, PCA and embedding compression. It also measures where
  the textbook Gram-Schmidt breaks down in floating point.

> Concept from BU DX601 (Math for Data Science), Week 10: vector spaces.

## What's inside (`subspaces.py`)

| Function | What it does |
|----------|--------------|
| `null_space(A)` | Exact basis for {x : Ax = 0}, one vector per free column, scaled to whole numbers |
| `equations_for_span(G)` | The n − m rows `a` with `a·g = 0` for every generator: span(G) = {x : Ax = 0} |
| `contains(equations, x)` | Is x in the subspace those equations cut out? |
| `intersect(eqs_U, eqs_W)` | Basis of U ∩ W: stack both sets of equations, take the null space |
| `dim_sum(gens_U, gens_W)` | dim(U + W), the rank of both generator sets together |
| `pivot_report(A)` | The read-off spelled out: pivot per row, spare rows, rank, free columns, dimension |
| `component_variances(X, center)` | PCA preview: variance along each principal direction |

## What's inside (`orthonormal.py`)

| Function | What it does |
|----------|--------------|
| `span_dimension(vectors)` | Dimension of the span = rank of the stacked vectors |
| `coordinates(x, B)` | The unique `c` with `B @ c == x` (solves a system) |
| `orthonormal_coordinates(x, Q)` | Same answer when `Q` is orthonormal: `Qᵀx`, just dot products |
| `gram_schmidt(vectors, method)` | Orthonormal columns `Q`, `"classical"` or `"modified"`; raises on dependent input |
| `proj_line(x, u)` | Closest point on the line through `u`: `(u·x / u·u) u` |
| `proj_subspace(x, Q)` | Closest point in span(`Q`): `Q (Qᵀx)` |
| `projection_matrix(A)` | Orthogonal projector onto span(`A`) for **any** basis: `A (AᵀA)⁻¹ Aᵀ` |
| `is_projection(P)` | `PP = P` |
| `is_orthogonal_projection(P)` | `PP = P` **and** `P = Pᵀ` |
| `orthogonality_loss(Q)` | `‖QᵀQ − I‖`, how far `Q` is from orthonormal |

## Run it

```bash
pip install -r requirements-dev.txt
python vector-spaces/subspaces.py
python vector-spaces/orthonormal.py
pytest vector-spaces
```

## Pivots decide everything

After elimination, the first non-zero entry in each row is a **pivot**. A pivot
column's unknown is *bound*: it gets solved in terms of the others. A column with no
pivot is *free*, and you can set it to anything. `pivot_report` spells out the read-off:

```
A = [[1, 2, 0, 3, 1], [2, 4, 1, 7, 1], [1, 2, 1, 4, 0]]
  row 1: first non-zero in column x1 -> pivot
  row 2: first non-zero in column x3 -> pivot
  row 3: all zeros -> a spare equation
  rank = 2 pivots; free columns: x2, x4, x5
  solution space dimension = columns - rank = 5 - 2 = 3
null space basis: (-2, 1, 0, 0, 0), (-3, 0, -1, 1, 0), (-1, 0, 1, 0, 1)
```

Each basis vector sets one free unknown to 1 and the others to 0, then reads every
pivot unknown off the RREF. Row 3 was row 3 = row 2 − row 1 in disguise, so it
added nothing. Only `Ax = 0` gives a subspace: `Ax = b` with b ≠ 0 is the same set
shifted off the origin, which can't contain 0.

## Going the other way: from a span to equations

An m-dimensional subspace of ℝⁿ needs **n − m** equations. Each equation's row `a`
must satisfy `a·g = 0` for every generator g, which is a homogeneous system with the
generators as rows, so the equations *are* a null space:

```
span of [[3, 1, -2], [1, 4, 5]] is cut out by [[13, -17, 11]]   np.cross: [13, -17, 11]
plane ∩ hyperplane in R^4: (-1, 1, -1, 1)  (dimension 1)
```

In 3-D the single equation is the cross product of the two generators, and the tests
check that. Any multiple works too, so `as_integers` picks the smallest whole-number
one. To intersect two subspaces, stack both sets of equations, since a point is in
both exactly when it satisfies both. The tests check the result against
dim(U ∩ W) = dim U + dim W − dim(U + W) on random subspaces.

## Data in a subspace: a first look at PCA

Stack data points as rows and the rank is the smallest subspace that holds them all.
PCA finds orthonormal directions ordered by how much the data spreads along them, so
data on an m-dimensional subspace of ℝⁿ has n − m directions with **zero** variance:

```
component variances, data on a 2-D plane in R^4 (expect 4 - 2 = 2 zeros)
  exact                 [5.935  1.6272 0.     0.    ]
  noisy                 [5.9339 1.6294 0.0001 0.0001]
  shifted, centered     [5.935  1.6272 0.     0.    ]
  shifted, NOT centered [51.645   4.8079  1.4712  0.    ]
```

With noise the extra directions are tiny rather than zero, which is how you spot "this
data is really 2-D". The last two rows show **why PCA subtracts the mean first**: a
plane that's been moved off the origin isn't a subspace, and measuring spread around
0 instead of around the data's center makes the offset look like a third direction.
Centering moves the plane back through the origin. Curved data, like a swiss roll, is
a *manifold*: no flat subspace fits it except locally.

## Why the ÷(u·u) disappears for a unit vector

Projecting x onto the line through u means finding the multiple `c u` whose residual
`x − c u` is perpendicular to u:

```
u · (x − c u) = 0   →   c = (u·x) / (u·u)
```

The `u·u` is there to undo u's length, which shows up twice: once in the dot product
`u·x` and once more when you multiply back by u. For a unit vector q, `q·q = 1`, so the
division is still there, it's just a division by 1. The general formula keeps the same
shape: `projection_matrix(A) = A (AᵀA)⁻¹ Aᵀ`, and for orthonormal columns `AᵀA = I`, so
it collapses to `Q Qᵀ`. The tests check that both routes give the same matrix.

## Slanted vs orthogonal: why `PP = P` isn't enough

Any matrix with `PP = P` is a projection: it moves vectors into a subspace and leaves
points already there alone. Only an **orthogonal** projection moves them to the
*closest* point, and the extra condition is symmetry, `P = Pᵀ`:

```
   slanted: PP=P True, orthogonal False, sends (0, 2) to [0. 0.]
orthogonal: PP=P True, orthogonal True,  sends (0, 2) to [1. 1.]
```

Both land on the line y = x, but `[[1,0],[1,0]]` slides points along a slanted
direction, while `u uᵀ / u·u` drops them straight down. Symmetry makes the residual
`(I − P)x` perpendicular to everything `P` produces, and a perpendicular residual is
what makes `Px` the closest point (Pythagoras).

## Classical vs modified Gram-Schmidt

In exact arithmetic they're the same algorithm. Classical computes every shadow from
the *original* vector, `(v·q) q`; modified removes each shadow from the *running*
remainder, `(u·q) q`, so the rounding error from earlier steps gets cleaned up instead
of carried along. On Hilbert matrices, whose columns are nearly parallel:

```
 n   cond(H)    classical   modified      ‖QᵀQ − I‖
 4   1.6e+04   4.9e-11     4.2e-13
 6   1.5e+07   2.4e-04     4.5e-10
 8   1.5e+10   1.4e+00     2.0e-07        <- classical Q is no longer orthogonal at all
10   1.6e+13   3.5e+00     1.6e-04
```

Classical's error grows roughly with cond(H)², modified's with cond(H). That's why
real libraries use modified Gram-Schmidt or, more often, Householder QR
(`np.linalg.qr`), whose loss stays near machine precision regardless of conditioning.

## Tests

`tests/test_subspaces.py` (64 tests) checks that every null-space vector solves
`Ax = 0` and the basis size is columns − rank (on fixed and random low-rank integer
matrices), that span equations vanish on every generator and combination and match
`np.cross` in 3-D, that span → equations → null space recovers the span, the
intersection dimension formula, the pivot read-off, and the PCA zero-variance counts
with and without noise and centering.

`tests/test_orthonormal.py` (28 tests) checks `QᵀQ = I` for both methods across shapes,
that the first j `q`'s span the same space as the first j inputs, that dependent input
raises, residuals are perpendicular, the projection beats 200 random points of the
subspace, the any-basis and orthonormal projectors agree, the slanted matrix is
idempotent but not orthogonal, and that modified beats classical by 10⁴ on Hilbert(8).
