import numpy as np
import pytest
from orthonormal import (
    coordinates,
    gram_schmidt,
    hilbert,
    is_orthogonal_projection,
    is_projection,
    orthogonality_loss,
    orthonormal_coordinates,
    proj_line,
    proj_subspace,
    projection_matrix,
    span_dimension,
)

rng = np.random.default_rng(601)


def random_vectors(k, n):
    return list(rng.normal(size=(k, n)))


# --- spans, bases, coordinates -------------------------------------------------

def test_span_dimension_counts_independent_directions():
    assert span_dimension([(1, 0, 2), (0, 3, 1), (2, 3, 5)]) == 2  # third = 2·first + second
    assert span_dimension([(1, 0, 2), (0, 3, 1), (0, 0, 1)]) == 3


def test_coordinates_rebuild_the_vector():
    B = np.column_stack([(2, 1, 0), (0, 1, 3), (1, 0, 1)]).astype(float)
    x = np.array([5.0, -2.0, 7.0])
    c = coordinates(x, B)
    assert np.allclose(B @ c, x)


def test_orthonormal_coordinates_are_dot_products_and_match_solve():
    Q = gram_schmidt(random_vectors(4, 4))
    x = rng.normal(size=4)
    assert np.allclose(orthonormal_coordinates(x, Q), coordinates(x, Q))


# --- Gram-Schmidt --------------------------------------------------------------

@pytest.mark.parametrize("method", ["classical", "modified"])
@pytest.mark.parametrize("k, n", [(1, 3), (2, 3), (3, 3), (3, 6), (5, 5)])
def test_gram_schmidt_gives_orthonormal_columns(method, k, n):
    Q = gram_schmidt(random_vectors(k, n), method)
    assert Q.shape == (n, k)
    assert np.allclose(Q.T @ Q, np.eye(k))


def test_gram_schmidt_keeps_the_same_span_in_order():
    vs = random_vectors(3, 5)
    Q = gram_schmidt(vs)
    for j in range(1, 4):
        # first j q's span the same space as the first j v's
        assert span_dimension([*vs[:j], *Q[:, :j].T]) == j


def test_first_vector_is_just_normalized():
    Q = gram_schmidt([(0, 6, 8), (1, 0, 0)])
    assert np.allclose(Q[:, 0], [0, 0.6, 0.8])


def test_dependent_input_raises():
    with pytest.raises(ValueError, match="dependent"):
        gram_schmidt([(1, 2, 3), (2, 4, 6)])
    with pytest.raises(ValueError, match="dependent"):
        gram_schmidt([(1, 0, 1), (0, 1, 1), (1, 1, 2)])


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        gram_schmidt([(1, 0)], method="householder")


def test_modified_loses_far_less_orthogonality_than_classical():
    cols = list(hilbert(8).T)
    classical = orthogonality_loss(gram_schmidt(cols, "classical", tol=0))
    modified = orthogonality_loss(gram_schmidt(cols, "modified", tol=0))
    assert classical > 0.1          # classical has visibly broken down
    assert modified < 1e-5          # modified is still close to orthonormal
    assert modified < classical / 1e4


# --- projections ---------------------------------------------------------------

def test_proj_line_residual_is_perpendicular():
    x, u = np.array([2.0, -1.0, 5.0]), np.array([1.0, 2.0, 2.0])
    p = proj_line(x, u)
    assert np.isclose((x - p) @ u, 0)
    assert np.allclose(p, (u @ x) / 9 * u)  # u·u = 9


def test_proj_line_ignores_the_length_of_u():
    x, u = rng.normal(size=4), rng.normal(size=4)
    assert np.allclose(proj_line(x, u), proj_line(x, -7.5 * u))


def test_unit_vector_drops_the_division():
    x, u = rng.normal(size=3), rng.normal(size=3)
    q = u / np.linalg.norm(u)
    assert np.allclose(proj_line(x, u), (q @ x) * q)


def test_proj_subspace_is_the_closest_point():
    Q = gram_schmidt(random_vectors(2, 4))
    x = rng.normal(size=4)
    p = proj_subspace(x, Q)
    assert np.allclose(Q.T @ (x - p), 0)
    best = np.linalg.norm(x - p)
    for _ in range(200):  # no other point of the subspace is closer
        other = Q @ rng.normal(size=2)
        assert np.linalg.norm(x - other) >= best - 1e-12


def test_projection_of_a_point_already_inside_is_itself():
    Q = gram_schmidt(random_vectors(2, 3))
    inside = Q @ np.array([1.5, -0.5])
    assert np.allclose(proj_subspace(inside, Q), inside)


def test_projection_matrix_any_basis_matches_orthonormal_route():
    vs = random_vectors(2, 4)
    A = np.column_stack(vs)
    Q = gram_schmidt(vs)
    assert np.allclose(projection_matrix(A), Q @ Q.T)


def test_projection_matrix_for_a_line_is_u_ut_over_utu():
    u = np.array([2.0, 1.0, 2.0])
    assert np.allclose(projection_matrix(u), np.outer(u, u) / 9)


def test_orthogonal_projection_is_idempotent_and_symmetric():
    P = projection_matrix(np.column_stack(random_vectors(2, 5)))
    assert is_projection(P)
    assert is_orthogonal_projection(P)


def test_slanted_projection_is_idempotent_but_not_orthogonal():
    S = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert is_projection(S)
    assert not is_orthogonal_projection(S)
    x = np.array([0.0, 2.0])
    straight = projection_matrix([1, 1]) @ x
    assert np.allclose(S @ x, [0, 0])
    assert np.allclose(straight, [1, 1])
    assert np.linalg.norm(x - straight) < np.linalg.norm(x - S @ x)


def test_non_projection_fails():
    R = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90° rotation
    assert not is_projection(R)
