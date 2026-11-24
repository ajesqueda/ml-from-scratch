# PCA, random projections and LDA

Principal components analysis finds the directions where data spreads out most. It's
the eigenvectors of the covariance matrix, so it builds directly on
[`eigen/`](../eigen/). This folder implements PCA from the covariance matrix, checks
the facts that should always hold on random data, and puts two neighbors next to it:
random projection (no training at all) and LDA (uses labels to separate two classes).

> Concept from BU DX601 (Math for Data Science), Week 13: principal components analysis.

## What's inside (`components.py`)

| Function | What it does |
|----------|--------------|
| `fit(X, standardize=False)` | Center, `C = XcᵀXc / n`, eigenvectors of C biggest-variance first |
| `PCA.directions` / `PCA.variances` | The principal components (unit vectors) and their λ, kept as separate fields |
| `PCA.components_for(target)` | Smallest k whose running variance share reaches the target |
| `PCA.coordinates(X)` | `cᵢ = wᵢ · (x − μ) / s` for every row |
| `PCA.reconstruct(Z, k)` | `μ + s ∘ (c₁w₁ + … + cₖwₖ)` |
| `random_projection(d, k, rng)` | Random normal k×d matrix scaled by 1/√k |
| `lda(X0, X1)` | `w = Σ⁻¹(μ₁ − μ₀)` and threshold `c = ½·w·(μ₁ + μ₀)` |

`s` is each column's standard deviation for `fit(X, standardize=True)` and 1 otherwise,
so without standardizing these are the plain formulas from the notes. `∘` multiplies
entry by entry.

## Run it

```bash
pip install -r requirements-dev.txt
python pca/components.py
pytest pca
```

```
variances λ: [6.419 0.087 0.085]  shares: [0.974 0.013 0.013]
components for 90%: 1
rebuild from PC1: squared error 34.584 = dropped coords² 34.584
height in      metres: PC1 (height, weight) = [0.006 1.   ], share 100%
height in centimetres: PC1 (height, weight) = [0.734 0.679], share 80%
standardized:           PC1 (height, weight) = [0.707 0.707], share 80%

random projection 50->10: squared-distance ratio mean 1.040, range 0.22-2.51

LDA direction [ 0.729 -0.685]: accuracy 83.0%
connect-the-means [ 0.999 -0.035]: accuracy 67.1%
```

## PC1 is a direction, λ₁ is a number

`fit` returns them as separate fields on purpose:

- `directions[:, 0]` is **PC1**: a unit vector, like (2, 1)/√5. It says *which way* the
  data spreads most.
- `variances[0]` is **λ₁**: a single number. It says *how much* the data spreads that way.

They come from the same eigen-pair, `C w₁ = λ₁ w₁`, and the variance of the data's
coordinates along w₁ is exactly λ₁. When an answer asks for "the first principal
component", give the vector; when it asks for "the variance it explains", give λ₁ (or
its share λ₁ / Σλ).

## Finding the eigenvectors of C

Building `C − λI` means subtracting λ **from the diagonal only**: λI has λ on the
diagonal and 0 everywhere else. After that, the rows of `C − λI` are always multiples
of each other (that's what det = 0 means), so there's one equation `a·x₁ + b·x₂ = 0`.
For a 2×2 C, instead of choosing a free variable, use the swap-and-negate rule from
[`eigen/`](../eigen/#step-2-eigenvectors-and-which-variable-to-set-to-1): **v = (−b, a)**.
It never divides, so there's no row-1-vs-row-2 decision. If row 1 is all zeros, use
row 2 the same way. Then divide by the length to make it a unit vector.

Two caveats. If both rows are zero, C = λI: the data spreads equally in every
direction and *any* unit vector is a principal component. And with more than two
features, det = 0 only says the rows are dependent, not that each is a multiple of the
first, so you need the full null space of `C − λI` (see
[`vector-spaces/`](../vector-spaces/)), which is what `eigh` does numerically.

`np.linalg.eigh` does the same thing numerically but returns eigenvalues in
**ascending** order, which is why `fit` flips them. It also picks each vector's sign
arbitrarily, so `fit` makes each direction's largest entry positive to keep answers
comparable.

## The invariants (checked on random data)

These should hold for any dataset, so the tests check them on three random correlated
datasets rather than on one worked example:

| Fact | Why |
|------|-----|
| `C` equals `np.cov(X.T, bias=True)` | `bias=True` divides by n, like the course |
| `C w = λ w` for every direction | they're eigen-pairs |
| `wᵀ C w = λ` = variance of the coordinates | λ is the spread along w |
| directions are orthonormal: `WᵀW = I` | C is symmetric |
| `Σλ = trace(C)` | total variance is the same in any orthonormal basis |
| `λᵢ = σᵢ² / n` from the SVD of the centered data | `XcᵀXc = V Σ² Vᵀ` |
| reconstruction error from k PCs = squared dropped coordinates = n·(dropped λ's) | the dropped pieces are perpendicular to the kept ones (measured in the standardized units when `standardize=True`) |

## How many components

Each component's share is `λᵢ / Σλ`. Add the shares in order and keep the smallest k
whose running total reaches the target. For λ = (50, 30, 12, 5, 3), the running shares
are 0.50, 0.80, 0.92, 0.97, 1.00, so 90% needs **k = 3**: the answer is a count, not
the share.

## Units matter: standardize first

PCA chases variance, and variance depends on units. In the demo, the same people are
measured with height in metres and weight in kilograms. Height's numbers barely move
(±0.08), weight's move by ±7, so PC1 is just "weight" with a 100% share. Measuring
height in centimetres instead makes its numbers bigger, and suddenly PC1 is a mix of
both. Nothing about the people changed, only the units.

`fit(X, standardize=True)` divides every column by its standard deviation first, and
then metres and centimetres give identical components. The tests check that.

## Random projection

Multiply by a random matrix with normal entries scaled by 1/√k. It needs no training
data, and squared distances survive **on average**: each row r contributes `(r·x)²`,
and its expected value is `‖x‖² / k`. Any single projection can stretch or squeeze a
particular pair, like the 8 vs 18 example from the notes. Over many random matrices
the ratio averages 1, and more output dimensions mean less distortion. Use it when the
data is huge and speed matters more than finding the best directions.

## LDA: using the labels

PCA ignores labels. LDA uses them to find one direction that separates two classes:

```
w = Σ⁻¹ (μ₁ − μ₀)          threshold  c = ½ · w · (μ₁ + μ₀)          class 1 if w·x > c
```

- **Which mean gets subtracted:** `μ₁ − μ₀` makes w point from class 0 toward class 1,
  so "bigger than the threshold" means class 1. Swapping the classes flips both w and c,
  and every prediction stays the same. The tests check that.
- **Why Σ⁻¹ changes the direction:** in the demo, each class's noise runs mostly along
  (1, 1), and the means differ along (1, 0). Just connecting the means points along
  (1, 0), right through the noise, and gets 67% right. Σ⁻¹ turns w away from the noisy
  direction to where the classes are tight, nearly perpendicular to (1, 1), and gets
  83%. When the noise is the same in every direction, Σ⁻¹ changes nothing and LDA is
  just "connect the means". The tests check both cases.

## Tests

`tests/test_components.py` (62 tests): every invariant in the table on three random
datasets; deterministic signs; rebuilding with every k; `components_for` against the
running shares; data on a plane needing exactly two components; units deciding PC1
until standardizing, and standardized models rebuilding original units; random
projections keeping squared distances on average with less spread for bigger k; and
LDA beating connect-the-means under correlated noise, pointing toward class 1,
flipping cleanly when the classes are swapped, and matching connect-the-means for
round noise; plus bad input: data with no variance, impossible k,
duplicate rows in the distance ratios, and LDA with a column that copies another.
