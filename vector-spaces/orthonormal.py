"""Orthonormal bases, Gram-Schmidt and projections, from scratch with NumPy.

Vectors go in as a list of 1-D sequences; bases come back as matrices whose
*columns* are the basis vectors, so Q.T @ x gives coordinates and Q @ c rebuilds x.
"""

import numpy as np


def span_dimension(vectors) -> int:
    """Dimension of span(vectors): stack them as rows and take the rank."""
    return int(np.linalg.matrix_rank(np.array(vectors, dtype=float)))


def coordinates(x, basis) -> np.ndarray:
    """The one recipe c with basis @ c == x (basis vectors as columns). Solves a system."""
    return np.linalg.solve(np.asarray(basis, dtype=float), np.asarray(x, dtype=float))


def orthonormal_coordinates(x, Q) -> np.ndarray:
    """Same answer as coordinates() when Q's columns are orthonormal: just dot products."""
    return np.asarray(Q, dtype=float).T @ np.asarray(x, dtype=float)


def gram_schmidt(vectors, method: str = "modified", tol: float = 1e-10) -> np.ndarray:
    """Orthonormalize vectors in order; returns Q with orthonormal columns.

    "classical" computes every shadow from the *original* v. "modified" removes each
    shadow from the running remainder u, which is the same math in exact arithmetic
    but loses far less orthogonality in floating point (see orthogonality_loss).
    """
    if method not in ("classical", "modified"):
        raise ValueError(f"method must be 'classical' or 'modified', got {method!r}")
    qs: list[np.ndarray] = []
    for v in vectors:
        v = np.asarray(v, dtype=float)
        u = v.copy()
        for q in qs:
            u = u - ((u if method == "modified" else v) @ q) * q  # remove the shadow on q
        norm = np.linalg.norm(u)
        if norm < tol * max(1.0, np.linalg.norm(v)):
            raise ValueError("vectors are linearly dependent (nothing left after removing shadows)")
        qs.append(u / norm)
    return np.column_stack(qs)


def proj_line(x, u) -> np.ndarray:
    """Closest point to x on the line through u: (u·x / u·u) u."""
    x, u = np.asarray(x, dtype=float), np.asarray(u, dtype=float)
    return (u @ x) / (u @ u) * u


def proj_subspace(x, Q) -> np.ndarray:
    """Closest point to x in the span of Q's orthonormal columns: Q (Qᵀx)."""
    Q = np.asarray(Q, dtype=float)
    return Q @ (Q.T @ np.asarray(x, dtype=float))


def projection_matrix(basis) -> np.ndarray:
    """Orthogonal projection onto span(basis columns), for ANY basis: A (AᵀA)⁻¹ Aᵀ.

    For one column u this is u uᵀ / (u·u), the line formula. For orthonormal columns
    AᵀA = I, so it collapses to Q Qᵀ: the division is still there, it's just by 1.
    """
    A = np.asarray(basis, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    return A @ np.linalg.solve(A.T @ A, A.T)


def is_projection(P, tol: float = 1e-10) -> bool:
    """Idempotent: projecting twice changes nothing (PP = P)."""
    P = np.asarray(P, dtype=float)
    return bool(np.allclose(P @ P, P, atol=tol))


def is_orthogonal_projection(P, tol: float = 1e-10) -> bool:
    """PP = P *and* P = Pᵀ. Symmetry is what makes the residual perpendicular,
    so P x is the closest point instead of a slanted shadow."""
    P = np.asarray(P, dtype=float)
    return is_projection(P, tol) and bool(np.allclose(P, P.T, atol=tol))


def orthogonality_loss(Q) -> float:
    """‖QᵀQ − I‖: 0 for a perfectly orthonormal Q."""
    Q = np.asarray(Q, dtype=float)
    return float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])))


def hilbert(n: int) -> np.ndarray:
    """H[i, j] = 1 / (i + j + 1): columns that are nearly parallel."""
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    Q = gram_schmidt([(1, 2, 2), (2, 0, 1)])
    print("Q =\n", Q, "\nQᵀQ =\n", Q.T @ Q)

    x = np.array([3.0, 1.0, 4.0])
    p = proj_subspace(x, Q)
    print("closest point:", p, " residual · columns:", (x - p) @ Q)

    slanted = np.array([[1.0, 0.0], [1.0, 0.0]])
    straight = projection_matrix([1, 1])
    for name, P in [("slanted", slanted), ("orthogonal", straight)]:
        print(f"{name:>10}: PP=P {is_projection(P)}, orthogonal {is_orthogonal_projection(P)},"
              f" sends (0, 2) to {P @ [0, 2]}")

    print("\northogonality loss ‖QᵀQ − I‖ on Hilbert matrices")
    print(" n   cond(H)    classical   modified")
    for n in (4, 6, 8, 10):
        H = hilbert(n)
        cols = list(H.T)
        cgs = orthogonality_loss(gram_schmidt(cols, "classical", tol=0))
        mgs = orthogonality_loss(gram_schmidt(cols, "modified", tol=0))
        print(f"{n:2d}   {np.linalg.cond(H):.1e}   {cgs:.1e}     {mgs:.1e}")
