import math

import numpy as np
import pytest
from vectors import (
    cosine_similarity,
    dot,
    euclidean_distance,
    l1_norm,
    l2_norm,
    max_norm,
    nearest_neighbor,
    np_cosine_similarity_matrix,
    np_nearest_neighbors,
    np_pairwise_distances,
    standardize,
    unit_vector,
)


def test_norms_of_a_3_4_vector():
    assert l2_norm([3, 4]) == 5
    assert l1_norm([3, -4]) == 7
    assert max_norm([3, -4]) == 4


@pytest.mark.parametrize("v", [[1.0, 2.0, 2.0], [-3.0, 0.5], [7.0]])
def test_norms_match_numpy(v):
    assert l1_norm(v) == pytest.approx(np.linalg.norm(v, 1))
    assert l2_norm(v) == pytest.approx(np.linalg.norm(v))
    assert max_norm(v) == pytest.approx(np.linalg.norm(v, np.inf))


def test_norm_ordering_holds():
    # For any vector: max <= L2 <= L1.
    rng = np.random.default_rng(0)
    for v in rng.normal(size=(50, 6)).tolist():
        assert max_norm(v) <= l2_norm(v) + 1e-12 <= l1_norm(v) + 1e-12


def test_unit_vector_has_length_one_and_same_direction():
    u = unit_vector([3, 4])
    assert u == pytest.approx([0.6, 0.8])
    assert l2_norm(u) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        unit_vector([0, 0])


def test_distance_is_norm_of_difference_and_symmetric():
    a, b = [1, 1], [4, 5]
    assert euclidean_distance(a, b) == 5
    assert euclidean_distance(a, b) == euclidean_distance(b, a)
    assert euclidean_distance(a, a) == 0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="different lengths"):
        dot([1, 2], [1, 2, 3])


@pytest.mark.parametrize(
    "a,b,expected",
    [([1, 0], [5, 0], 1.0), ([1, 0], [0, 3], 0.0), ([1, 2], [-2, -4], -1.0)],
)
def test_cosine_similarity_depends_only_on_direction(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected)


def test_cosine_is_cos_of_angle():
    assert cosine_similarity([1, 0], [1, 1]) == pytest.approx(math.cos(math.pi / 4))


def test_nearest_neighbor():
    points = [[0, 0], [10, 10], [3, 4]]
    assert nearest_neighbor([2, 3], points) == 2
    with pytest.raises(ValueError):
        nearest_neighbor([0, 0], [])


def test_numpy_versions_match_pure_python():
    rng = np.random.default_rng(1)
    points, queries = rng.normal(size=(40, 5)), rng.normal(size=(10, 5))

    dists = np_pairwise_distances(points, queries)
    assert dists.shape == (10, 40)
    assert dists[3, 7] == pytest.approx(euclidean_distance(queries[3], points[7]))

    expected = [nearest_neighbor(q.tolist(), points.tolist()) for q in queries]
    assert np_nearest_neighbors(points, queries).tolist() == expected

    cos = np_cosine_similarity_matrix(queries, points)
    assert cos[2, 5] == pytest.approx(cosine_similarity(queries[2], points[5]))


def test_scaling_changes_the_nearest_neighbor():
    houses = np.array([[2000.0, 2], [2050.0, 5], [2300.0, 3]])
    query = np.array([[2040.0, 3]])
    assert np_nearest_neighbors(houses, query)[0] == 1

    scaled, mean, std = standardize(houses)
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0), 1)
    assert np_nearest_neighbors(scaled, (query - mean) / std)[0] == 0
