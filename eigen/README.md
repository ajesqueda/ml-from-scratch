# Eigenvalues and eigenvectors by hand (2×2)

An eigenvector is an arrow a matrix only stretches, shrinks or flips, never turns:
`A v = λ v`, with λ the stretch factor and v never the zero vector. This folder finds
them the way you would on paper, exactly with fractions when the answer is clean,
handles the cases that usually trip people up, and previews why the biggest
eigenvalue matters through power iteration.

> Concept from BU DX601 (Math for Data Science), Week 12: eigenvectors and matrix decompositions.

## What's inside (`eigen_by_hand.py`)

| Function | What it does |
|----------|--------------|
| `eig2x2(A)` | Both eigenvalues from trace and det, biggest \|λ\| first; exact when the square root is rational |
| `eigenvector(A, λ)` | Solves `(A − λI)x = 0`; rejects a λ that isn't an eigenvalue |
| `eigenpairs(A)` | Every distinct eigenvalue with its eigenvectors, including repeated eigenvalues |
| `power_iteration(A)` | λ₁ and its eigenvector by multiplying by A over and over; raises if it never settles (two \|λ\| tie, or a complex pair) |
| `trace_det(A)` | The two numbers that decide everything for a 2×2 |

## Run it

```bash
pip install -r requirements-dev.txt
python eigen/eigen_by_hand.py
pytest eigen
```

```
two clean eigenvalues: A = [[4, 1], [2, 3]], trace 7, det 10
  λ = 5: (1, 1)
  λ = 2: (1, -2)
negative λ₁ (sign flip drill): A = [[-4, 3], [2, 1]], trace -3, det -10
  λ = -5: (3, -1)
  λ = 2: (1, 2)
irrational eigenvalues: A = [[2, 1], [1, 1]], trace 3, det 1
  λ = 2.6180: (0.8507, 0.5257)
  λ = 0.3820: (0.5257, -0.8507)
repeated, only one eigenvector (shear): A = [[3, 1], [0, 3]], trace 6, det 9
  λ = 3: (1, 0)
repeated, every vector (A = 3I): A = [[3, 0], [0, 3]], trace 6, det 9
  λ = 3: (1, 0), (0, 1)
90° rotation [[0, -1], [1, 0]]: m² − p = -1 < 0: no real eigenvalues
power iteration on [[4, 1], [2, 3]]: λ₁ ≈ 5.000000, v ≈ [0.7071 0.7071]
power iteration on [[-4, 3], [2, 1]]: λ₁ ≈ -5.000000, v ≈ [ 0.9487 -0.3162]
```

## Step 1: eigenvalues from trace and determinant

For λ to have a non-zero solution of `(λI − A)v = 0`, the matrix `λI − A` must be
singular, so `det(λI − A) = 0`. For a 2×2 that always works out to

```
λ² − (trace) λ + det = 0
```

With `m = trace / 2` and `p = det`, the quadratic formula collapses to
**λ = m ± √(m² − p)**. Two checks come for free: the eigenvalues add up to the trace
and multiply to the determinant, and the tests confirm both on random matrices. A
triangular matrix's eigenvalues are just its diagonal, because its determinant is
the product of the diagonal.

**The sign flip.** When you factor instead, a factor `(λ + r)` is zero at λ = **−r**:

| Factored | Eigenvalues |
|----------|-------------|
| (λ + 1)(λ + 2) = 0 | −1, −2 |
| (λ + 5)(λ − 2) = 0 | −5, 2 |
| (λ − 3)(λ − 4) = 0 | 3, 4 |

"Multiply to det, add to trace" works on the eigenvalues themselves, not on the
numbers in the factors: −5 · 2 = −10 = det and −5 + 2 = −3 = trace for
`[[-4, 3], [2, 1]]`.

λ₁ is the eigenvalue with the biggest |λ|, and it keeps its sign: here λ₁ = −5, not 2.

## Step 2: eigenvectors, and which variable to set to 1

Subtract λ from the **diagonal only** (`λI` has λ on the diagonal and 0 elsewhere).
The rows of `A − λI` are then multiples of each other, so there's really one equation:

```
a·x₁ + b·x₂ = 0
```

You don't have to choose a free variable. **Swap the two coefficients and negate one:
v = (−b, a).** It always works, because `a·(−b) + b·a = 0`, and it never divides.
Setting x₂ = 1 and solving gives the same line, just scaled:

```
λ = 2 for [[4, 1], [2, 3]]:  A − 2I = [[2, 1], [2, 1]]  ->  2·x₁ + 1·x₂ = 0
swap and negate:  v = (−1, 2)  ~  (1, −2)     setting x₂ = 1:  v = (−0.5, 1)  ~  (1, −2)
```

The answer is the **whole vector** (x₁, x₂), not just the variable you solved for,
and any non-zero multiple is the same eigenvector. `eigenvector` scales to the smallest
whole numbers with the first entry positive, so answers are easy to compare.

If the rows of `A − λI` are **not** multiples of each other, λ was wrong: the only
solution is x = 0. `eigenvector` checks that (det of `A − λI` must be 0) and raises.

## The cases that trip people up

- **No real eigenvalues.** When m² − p < 0, the matrix turns every arrow, so nothing is
  only stretched. A 90° rotation `[[0, -1], [1, 0]]` is the standard example.
- **A repeated eigenvalue with only one eigenvector.** The shear `[[3, 1], [0, 3]]` has
  λ = 3 twice, but `A − 3I = [[0, 1], [0, 0]]` only allows x₂ = 0, so (1, 0) is the
  only direction. A matrix like this can't be diagonalized.
- **A repeated eigenvalue where every vector works.** For `A = 3I`, `A − 3I` is all
  zeros, so every non-zero vector is an eigenvector.
- **A first row of zeros.** Then the equation lives in the second row; `eigenvector`
  falls through to it.

## Power iteration: why λ₁ matters

Write any start vector as a mix of the two eigenvectors. Each multiply by A scales the
λ₁ part by |λ₁| and the λ₂ part by |λ₂|, so after k steps the λ₂ part has shrunk by
|λ₂ / λ₁|ᵏ compared with the λ₁ part. Multiply enough times, rescaling so nothing
overflows, and the vector lines up with λ₁'s eigenvector. The Rayleigh quotient
`v·Av / v·v` then reads off λ₁ with its sign, even when λ₁ < 0 makes the vector flip
direction on every step.

That's the idea behind PageRank (the main eigenvector of the link matrix) and the
first principal component in PCA. The tests check the convergence rate: for
`[[4, 1], [2, 3]]` the error shrinks by at least (2/5)⁴ every five steps.

## Tests

`tests/test_eigen_by_hand.py` (175 tests, mostly over random integer matrices):

- **Eigenvalues:** exact clean examples, the sign-flip example, exact fractions and
  irrational floats; sum = trace and product = det; agreement with
  `np.linalg.eigvals`; triangular diagonals; rotations rejected.
- **Eigenvectors:** whole-number answers; `A v = λ v` and v ≠ 0 for every pair;
  parallel to NumPy's eigenvectors; wrong λ rejected; a zero first row; the shear's
  single eigenvector; A = 3I accepting every vector; unit length for irrational cases.
- **Power iteration:** finds λ₁ with its sign, including a negative λ₁, and the error
  shrinks at the |λ₂ / λ₁| rate.
