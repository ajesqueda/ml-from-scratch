# Vector spaces: orthonormal bases, Gram-Schmidt and projections

Orthonormal bases turn "solve a system" into "take dot products", and orthogonal
projection is the closest-point operation behind least squares, PCA and embedding
compression. This folder builds both from scratch with NumPy, tests the geometry,
and measures where the textbook Gram-Schmidt breaks down in floating point.

> Concept from BU DX601 (Math for Data Science), Week 10: vector spaces.

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
python vector-spaces/orthonormal.py
pytest vector-spaces
```

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

`tests/test_orthonormal.py` (28 tests) checks `QᵀQ = I` for both methods across shapes,
that the first j `q`'s span the same space as the first j inputs, that dependent input
raises, residuals are perpendicular, the projection beats 200 random points of the
subspace, the any-basis and orthonormal projectors agree, the slanted matrix is
idempotent but not orthogonal, and that modified beats classical by 10⁴ on Hilbert(8).
