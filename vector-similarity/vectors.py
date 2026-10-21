"""Vector norms, distances and nearest neighbors, in pure Python and NumPy.

These few functions sit underneath k-nearest neighbors, k-means, embedding
search and recommendation systems. Each one is written twice: a plain
Python loop that spells out the formula, and a NumPy version that is what
you'd use in practice. The tests check that they agree.

Run `python vectors.py` to see why feature scaling changes nearest neighbors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Vector = Sequence[float]


def _check_same_length(a: Vector, b: Vector) -> None:
    if len(a) != len(b):
        raise ValueError(f"vectors have different lengths: {len(a)} vs {len(b)}")


# --- Norms: "how big is this vector?" ------------------------------------


def l1_norm(v: Vector) -> float:
    """Sum of absolute values. Distance walking a city grid ("Manhattan")."""
    return sum(abs(x) for x in v)


def l2_norm(v: Vector) -> float:
    """Straight-line length: sqrt of the sum of squares (Pythagoras in n-D)."""
    return math.sqrt(sum(x * x for x in v))


def max_norm(v: Vector) -> float:
    """Largest absolute entry. Answers "what's the worst single coordinate?"."""
    return max(abs(x) for x in v)


def unit_vector(v: Vector) -> list[float]:
    """Same direction, length 1: a vector split into magnitude x direction."""
    length = l2_norm(v)
    if length == 0:
        raise ValueError("the zero vector has no direction")
    return [x / length for x in v]


# --- Distances and similarity: "how far apart / how alike?" ---------------


def euclidean_distance(a: Vector, b: Vector) -> float:
    # Distance is just the norm of the difference vector a - b.
    _check_same_length(a, b)
    return l2_norm([x - y for x, y in zip(a, b, strict=True)])


def dot(a: Vector, b: Vector) -> float:
    _check_same_length(a, b)
    return sum(x * y for x, y in zip(a, b, strict=True))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """cos(angle between a and b): 1 same direction, 0 unrelated, -1 opposite.

    It ignores length, which is why text and embedding search use it. A long
    document and a short one about the same topic point the same way.
    """
    denom = l2_norm(a) * l2_norm(b)
    if denom == 0:
        raise ValueError("cosine similarity is undefined for the zero vector")
    return dot(a, b) / denom


def nearest_neighbor(query: Vector, points: Sequence[Vector]) -> int:
    """Index of the point closest to `query` (Euclidean)."""
    if not points:
        raise ValueError("no points to search")
    distances = [euclidean_distance(query, p) for p in points]
    return min(range(len(points)), key=distances.__getitem__)


# --- The NumPy versions ----------------------------------------------------


def np_pairwise_distances(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """All query-to-point Euclidean distances at once, shape (n_queries, n_points).

    Broadcasting (q, 1, d) - (1, p, d) builds every difference vector, and
    norm over the last axis turns each into a length. No Python loop.
    """
    return np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=-1)


def np_nearest_neighbors(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return np_pairwise_distances(points, queries).argmin(axis=1)


def np_cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Normalize rows first; then cosine similarity is just a matrix product.
    a_unit = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_unit = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_unit @ b_unit.T


def standardize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rescale each column to mean 0, std 1. Returns (scaled, mean, std)."""
    mean, std = points.mean(axis=0), points.std(axis=0)
    return (points - mean) / std, mean, std


def main() -> None:
    # Houses as [square feet, bedrooms]. Square feet is ~1000x bigger in
    # magnitude, so unscaled Euclidean distance barely notices bedrooms.
    houses = np.array([[2000.0, 2], [2050.0, 5], [2300.0, 3]])
    query = np.array([[2040.0, 3]])
    print("houses (sqft, beds):", houses.tolist(), " query:", query.tolist()[0])

    raw = np_nearest_neighbors(houses, query)[0]
    scaled_houses, mean, std = standardize(houses)
    scaled = np_nearest_neighbors(scaled_houses, (query - mean) / std)[0]
    print(f"nearest without scaling: house {raw} {houses[raw].tolist()}  (sqft dominates)")
    print(f"nearest after scaling:   house {scaled} {houses[scaled].tolist()}")

    doc_short, doc_long = [1.0, 2.0, 0.0], [10.0, 20.0, 0.0]
    euclid = euclidean_distance(doc_short, doc_long)
    cosine = cosine_similarity(doc_short, doc_long)
    print(f"\nshort vs long doc, same topic: Euclidean = {euclid:.1f}, cosine = {cosine:.2f}")


if __name__ == "__main__":
    main()
