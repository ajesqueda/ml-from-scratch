"""PCA from the covariance matrix, random projections, and two-class LDA.

    fit(X)                     center, C = XcᵀXc / n, eigenvectors biggest-variance first
    PCA.components_for(0.9)    smallest k whose variance share reaches the target
    PCA.coordinates(X)         c_i = w_i · (x − μ) / s
    PCA.reconstruct(Z, k)      μ + s ∘ (c_1 w_1 + … + c_k w_k)
    random_projection(d, k)    a random k×d matrix scaled by 1/√k
    lda(X0, X1)                w = Σ⁻¹(μ1 − μ0) and the midpoint threshold

A fitted PCA keeps `directions` (the principal components, unit vectors) and
`variances` (the eigenvalues λ) as separate fields: PC1 is a direction, λ1 is how
much the data spreads along it. s is each column's standard deviation when fitted
with standardize=True, and 1 otherwise (then the formulas are the plain ones).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def covariance(X) -> np.ndarray:
    """C = XcᵀXc / n, dividing by n like the course (np.cov(X.T, bias=True))."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / len(X)


def _fix_signs(vecs: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """An eigenvector's sign is arbitrary; make each column's largest entry positive.

    Entries that tie for largest, like the two in (0.707, −0.707), differ only by
    rounding noise, so the first of them decides. Otherwise the sign could change
    from one machine to the next.
    """
    mags = np.abs(vecs)
    idx = np.argmax(mags >= mags.max(axis=0) - tol, axis=0)  # first near-largest entry
    signs = np.sign(vecs[idx, np.arange(vecs.shape[1])])
    return vecs * np.where(signs == 0, 1, signs)


@dataclass
class PCA:
    mean: np.ndarray
    scale: np.ndarray  # 1 for every column unless standardized
    directions: np.ndarray  # columns are the principal components, unit length
    variances: np.ndarray  # λ for each direction, biggest first

    def explained(self) -> np.ndarray:
        """Each component's share of the total variance: λ_i / Σλ."""
        total = self.variances.sum()
        if total == 0:
            raise ValueError("the data has no variance (every row is the same), "
                             "so no component explains any of it")
        return self.variances / total

    def components_for(self, target: float) -> int:
        """Smallest k whose running share reaches `target` (a count, not a share)."""
        if not 0 < target <= 1:
            raise ValueError("target is a fraction in (0, 1]")
        running = np.cumsum(self.explained())
        return int(np.argmax(running >= target - 1e-12) + 1)

    def coordinates(self, X) -> np.ndarray:
        """Each row's coordinates along the PCs: w_i · (x − μ) / s for every row."""
        Z = (np.asarray(X, dtype=float) - self.mean) / self.scale
        return Z @ self.directions

    def reconstruct(self, coords, k: int | None = None) -> np.ndarray:
        """Rebuild rows from their first k coordinates (all of them by default).

        μ + s ∘ (c_1 w_1 + … + c_k w_k): rebuilt in the standardized space, then
        scaled back to the original units.
        """
        coords = np.atleast_2d(coords)
        k = coords.shape[1] if k is None else k
        if not 0 <= k <= min(coords.shape[1], self.directions.shape[1]):
            raise ValueError(f"k = {k}: need 0 <= k <= {coords.shape[1]} components")
        rebuilt = coords[:, :k] @ self.directions[:, :k].T
        return rebuilt * self.scale + self.mean


def fit(X, standardize: bool = False) -> PCA:
    """Principal components of the rows of X.

    With standardize=True every column is first divided by its standard deviation,
    so a column measured in big units (grams, dollars) can't take over PC1 just
    because its numbers are bigger.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0) if standardize else np.ones(X.shape[1])
    if np.any(scale == 0):
        raise ValueError("a constant column can't be standardized")
    vals, vecs = np.linalg.eigh(covariance((X - mean) / scale))  # eigh: ascending order
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0, None)  # tiny negatives are rounding error
    return PCA(mean, scale, _fix_signs(vecs[:, order]), vals)


def random_projection(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k×d matrix of standard normal entries scaled by 1/√k.

    Each squared length is kept on average: E‖Rx‖² = ‖x‖², because every row of R
    contributes (r·x)² with E[(r·x)²] = ‖x‖² / k. No training needed.
    """
    if d < 1 or k < 1:
        raise ValueError("projecting needs d >= 1 input and k >= 1 output dimensions")
    return rng.standard_normal((k, d)) / np.sqrt(k)


def squared_distance_ratios(X, R) -> np.ndarray:
    """‖R(x − y)‖² / ‖x − y‖² for every pair of distinct rows.

    Identical rows are skipped: their distance is 0 before and after, so the ratio
    0/0 says nothing about stretching.
    """
    X = np.asarray(X, dtype=float)
    i, j = np.triu_indices(len(X), k=1)
    diffs = X[i] - X[j]
    diffs = diffs[(diffs**2).sum(axis=1) > 0]
    return ((diffs @ R.T) ** 2).sum(axis=1) / (diffs**2).sum(axis=1)


def lda(X0, X1) -> tuple[np.ndarray, float]:
    """Fisher's two-class direction and threshold.

    w = Σ⁻¹(μ1 − μ0), with Σ the pooled within-class covariance, and the threshold
    sits halfway between the class means along w: c = ½·w·(μ1 + μ0). The order
    μ1 − μ0 makes w point from class 0 toward class 1, so w·x > c means class 1.
    Swapping the classes flips both w and c, and the predictions stay the same.

    Without Σ⁻¹ the direction is just "connect the means". Σ⁻¹ turns it away from
    directions where each class is noisy, toward ones where the classes are tight.

    If Σ can't be inverted (a column that's a copy of another, or too few points),
    the pseudoinverse gives the shortest w that works, instead of failing.
    """
    X0, X1 = np.asarray(X0, dtype=float), np.asarray(X1, dtype=float)
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    pooled = (len(X0) * covariance(X0) + len(X1) * covariance(X1)) / (len(X0) + len(X1))
    w = np.linalg.pinv(pooled, hermitian=True) @ (mu1 - mu0)
    return w, float(0.5 * w @ (mu1 + mu0))


def predict(w: np.ndarray, c: float, X) -> np.ndarray:
    return (np.asarray(X, dtype=float) @ w > c).astype(int)


def accuracy(w, c, X0, X1) -> float:
    right = (predict(w, c, X0) == 0).sum() + (predict(w, c, X1) == 1).sum()
    return float(right / (len(X0) + len(X1)))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    rng = np.random.default_rng(13)

    # Data that mostly lies along one direction, plus a little noise.
    t = rng.normal(size=200)
    X = np.column_stack([2 * t, t, 0.5 * t]) + rng.normal(scale=0.3, size=(200, 3))
    model = fit(X)
    print("PC directions (columns):\n", model.directions)
    print("variances λ:", model.variances, " shares:", model.explained())
    print("components for 90%:", model.components_for(0.9))
    Z = model.coordinates(X)
    err = ((X - model.reconstruct(Z, 1)) ** 2).sum()
    dropped = (Z[:, 1:] ** 2).sum()
    print(f"rebuild from PC1: squared error {err:.3f} = dropped coords² {dropped:.3f}")

    # Units: the same people, height in metres or centimetres, weight in kg.
    height_m = rng.normal(1.75, 0.08, size=300)
    weight_kg = 70 + 60 * (height_m - 1.75) + rng.normal(0, 6, size=300)
    for label, H in [("metres", height_m), ("centimetres", height_m * 100)]:
        model = fit(np.column_stack([H, weight_kg]))
        print(f"height in {label:>11}: PC1 (height, weight) = {model.directions[:, 0]}, "
              f"share {model.explained()[0]:.0%}")
    std = fit(np.column_stack([height_m, weight_kg]), standardize=True)
    print(f"standardized:           PC1 (height, weight) = {std.directions[:, 0]}, "
          f"share {std.explained()[0]:.0%}")

    # Random projection 50 -> 10 dimensions.
    pts = rng.normal(size=(40, 50))
    ratios = squared_distance_ratios(pts, random_projection(50, 10, rng))
    print(f"\nrandom projection 50->10: squared-distance ratio mean {ratios.mean():.3f}, "
          f"range {ratios.min():.2f}-{ratios.max():.2f}")

    # LDA vs connecting the means, with correlated noise inside each class.
    cov = np.array([[4.0, 3.6], [3.6, 4.0]])
    X0 = rng.multivariate_normal([0, 0], cov, size=500)
    X1 = rng.multivariate_normal([1.5, 0], cov, size=500)
    w, c = lda(X0, X1)
    naive = X1.mean(axis=0) - X0.mean(axis=0)
    c_naive = 0.5 * naive @ (X1.mean(axis=0) + X0.mean(axis=0))
    print(f"\nLDA direction {w / np.linalg.norm(w)}: accuracy {accuracy(w, c, X0, X1):.1%}")
    print(f"connect-the-means {naive / np.linalg.norm(naive)}: "
          f"accuracy {accuracy(naive, c_naive, X0, X1):.1%}")
