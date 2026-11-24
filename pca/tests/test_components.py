"""The invariants from pca_numpy_check.py, run on random data instead of one example."""

import numpy as np
import pytest
from components import (
    PCA,
    accuracy,
    covariance,
    fit,
    lda,
    predict,
    random_projection,
    squared_distance_ratios,
)

rng = np.random.default_rng(1322)


def correlated(n, d, seed):
    local = np.random.default_rng(seed)
    return local.normal(size=(n, d)) @ local.normal(size=(d, d)) + local.normal(size=d)


DATA = [correlated(n, d, seed) for seed, (n, d) in enumerate([(30, 2), (50, 3), (200, 5)])]


# --- the invariants --------------------------------------------------------------------

@pytest.mark.parametrize("X", DATA)
def test_covariance_divides_by_n(X):
    assert np.allclose(covariance(X), np.cov(X.T, bias=True))


@pytest.mark.parametrize("X", DATA)
def test_biggest_variance_first(X):
    lams = fit(X).variances
    assert np.all(np.diff(lams) <= 1e-12)


@pytest.mark.parametrize("X", DATA)
def test_each_direction_is_an_eigenvector_of_C(X):
    model, C = fit(X), covariance(X)
    for w, lam in zip(model.directions.T, model.variances, strict=True):
        assert np.allclose(C @ w, lam * w)


@pytest.mark.parametrize("X", DATA)
def test_lambda_is_the_variance_along_its_direction(X):
    model, C = fit(X), covariance(X)
    Xc = X - X.mean(axis=0)
    for w, lam in zip(model.directions.T, model.variances, strict=True):
        assert w @ C @ w == pytest.approx(lam)
        assert np.var(Xc @ w) == pytest.approx(lam, abs=1e-9)


@pytest.mark.parametrize("X", DATA)
def test_directions_are_orthonormal(X):
    W = fit(X).directions
    assert np.allclose(W.T @ W, np.eye(W.shape[1]))


@pytest.mark.parametrize("X", DATA)
def test_total_variance_is_the_trace(X):
    assert fit(X).variances.sum() == pytest.approx(np.trace(covariance(X)))


@pytest.mark.parametrize("X", DATA)
def test_lambda_is_sigma_squared_over_n(X):
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    assert np.allclose(s**2 / len(X), fit(X).variances)


@pytest.mark.parametrize("X", DATA)
def test_pcs_match_svd_directions_up_to_sign(X):
    _, _, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    for w, v in zip(fit(X).directions.T, Vt, strict=True):
        assert np.allclose(np.abs(w @ v), 1)


@pytest.mark.parametrize("X", DATA)
def test_reconstruction_error_is_the_dropped_variance(X):
    model = fit(X)
    Z = model.coordinates(X)
    d = X.shape[1]
    for k in range(d + 1):
        err = ((X - model.reconstruct(Z, k)) ** 2).sum()
        assert err == pytest.approx((Z[:, k:] ** 2).sum())
        assert err == pytest.approx(len(X) * model.variances[k:].sum())


@pytest.mark.parametrize("X", DATA)
def test_all_components_rebuild_exactly(X):
    model = fit(X)
    assert np.allclose(model.reconstruct(model.coordinates(X)), X)


def test_coordinates_are_dot_products_with_the_centered_row():
    X = DATA[1]
    model = fit(X)
    row = X[7]
    assert np.allclose(model.coordinates(row[None, :])[0],
                       [w @ (row - model.mean) for w in model.directions.T])


def test_tied_entries_use_the_first_one_for_the_sign():
    from components import _fix_signs

    v = np.array([[0.70710678, 0.70710678], [0.70710678000001, -0.70710678]])
    flipped = _fix_signs(-v)
    assert flipped[0, 0] > 0 and flipped[0, 1] > 0  # first entry positive, even in a tie
    assert np.allclose(_fix_signs(v), _fix_signs(-v))


def test_signs_are_deterministic():
    model = fit(DATA[2])
    big = np.argmax(np.abs(model.directions), axis=0)
    assert np.all(model.directions[big, np.arange(model.directions.shape[1])] > 0)
    assert np.allclose(fit(DATA[2]).directions, fit(DATA[2] * 1.0).directions)


# --- how many components -------------------------------------------------------------------

def make_model(lams):
    lams = np.array(lams, dtype=float)
    return PCA(np.zeros(len(lams)), np.ones(len(lams)), np.eye(len(lams)), lams)


@pytest.mark.parametrize(
    "target, k", [(0.5, 1), (0.8, 2), (0.9, 3), (0.92, 3), (0.95, 4), (1.0, 5)]
)
def test_components_for_target(target, k):
    model = make_model([50, 30, 12, 5, 3])
    assert model.components_for(target) == k  # running shares 0.50, 0.80, 0.92, 0.97, 1.00


def test_components_for_rejects_percent_numbers():
    with pytest.raises(ValueError):
        make_model([1, 1]).components_for(90)


def test_data_on_a_plane_needs_two_components():
    X = rng.normal(size=(100, 2)) @ rng.normal(size=(2, 4))
    model = fit(X)
    assert model.components_for(0.999999) == 2
    assert np.allclose(model.variances[2:], 0, atol=1e-9)


# --- units and standardizing -------------------------------------------------------------------

def test_units_decide_pc1_until_you_standardize():
    height_m = rng.normal(1.75, 0.08, size=400)
    weight_kg = 70 + 60 * (height_m - 1.75) + rng.normal(0, 6, size=400)
    in_m = fit(np.column_stack([height_m, weight_kg])).directions[:, 0]
    in_cm = fit(np.column_stack([height_m * 100, weight_kg])).directions[:, 0]
    assert abs(in_m[1]) > 0.99  # metres: weight's bigger numbers take over PC1
    assert abs(in_cm[0]) > 0.5  # centimetres: height counts too
    a = fit(np.column_stack([height_m, weight_kg]), standardize=True)
    b = fit(np.column_stack([height_m * 100, weight_kg]), standardize=True)
    assert np.allclose(a.directions, b.directions)  # standardized: units no longer matter
    assert np.allclose(a.variances, b.variances)


def test_standardized_model_still_rebuilds_original_units():
    X = DATA[1] * [1, 1000, 0.01]
    model = fit(X, standardize=True)
    assert np.allclose(model.reconstruct(model.coordinates(X)), X)


def test_constant_column_cannot_be_standardized():
    with pytest.raises(ValueError, match="constant"):
        fit(np.column_stack([np.arange(5.0), np.ones(5)]), standardize=True)


# --- random projection -----------------------------------------------------------------------

def test_random_projection_keeps_squared_distances_on_average():
    pts = rng.normal(size=(30, 60))
    means = [squared_distance_ratios(pts, random_projection(60, 15, rng)).mean()
             for _ in range(200)]
    assert np.mean(means) == pytest.approx(1, abs=0.03)


def test_more_dimensions_means_less_distortion():
    pts = rng.normal(size=(30, 200))

    def spread(k):
        return np.std(squared_distance_ratios(pts, random_projection(200, k, rng)))

    assert spread(100) < spread(25) < spread(5)


def test_one_fixed_projection_can_squeeze_a_pair():
    R = np.array([[1, -1, 0, 0], [0, 0, 1, 1]]) / np.sqrt(2)
    ratio = squared_distance_ratios(np.array([[4, 1, 0, 3], [1, 2, 2, 1]]), R)[0]
    assert ratio == pytest.approx(8 / 18)


# --- LDA ---------------------------------------------------------------------------------------

def two_classes(n=600, seed=0):
    local = np.random.default_rng(seed)
    cov = np.array([[4.0, 3.6], [3.6, 4.0]])  # noise mostly along (1, 1)
    return (local.multivariate_normal([0, 0], cov, size=n),
            local.multivariate_normal([1.5, 0], cov, size=n))


def test_lda_beats_connecting_the_means_with_correlated_noise():
    X0, X1 = two_classes()
    w, c = lda(X0, X1)
    naive = X1.mean(axis=0) - X0.mean(axis=0)
    c_naive = 0.5 * naive @ (X1.mean(axis=0) + X0.mean(axis=0))
    assert accuracy(w, c, X0, X1) > accuracy(naive, c_naive, X0, X1) + 0.1


def test_lda_direction_turns_away_from_the_noise():
    X0, X1 = two_classes()
    w, _ = lda(X0, X1)
    unit = w / np.linalg.norm(w)
    noise = np.array([1, 1]) / np.sqrt(2)
    assert abs(unit @ noise) < 0.2  # nearly perpendicular to the noisy direction


def test_w_points_toward_class_1():
    X0, X1 = two_classes()
    w, c = lda(X0, X1)
    assert w @ X1.mean(axis=0) > c > w @ X0.mean(axis=0)


def test_swapping_classes_flips_w_and_c_but_not_the_answers():
    X0, X1 = two_classes(seed=3)
    w, c = lda(X0, X1)
    w2, c2 = lda(X1, X0)
    assert np.allclose(w2, -w) and c2 == pytest.approx(-c)
    test = np.vstack([X0[:50], X1[:50]])
    assert np.array_equal(predict(w, c, test), 1 - predict(w2, c2, test))


def test_with_round_noise_lda_is_connecting_the_means():
    local = np.random.default_rng(9)
    X0 = local.normal(size=(5000, 2))
    X1 = local.normal(size=(5000, 2)) + [2, 1]
    w, _ = lda(X0, X1)
    naive = X1.mean(axis=0) - X0.mean(axis=0)
    cos = w @ naive / (np.linalg.norm(w) * np.linalg.norm(naive))
    assert cos > 0.999


# --- bad or degenerate input -----------------------------------------------------------

def test_data_with_no_variance_says_so_instead_of_claiming_a_component():
    model = fit(np.ones((5, 3)))
    with pytest.raises(ValueError, match="no variance"):
        model.explained()
    with pytest.raises(ValueError, match="no variance"):
        model.components_for(0.9)


@pytest.mark.parametrize("k", [-1, 4])
def test_reconstruct_rejects_impossible_k(k):
    model = fit(DATA[1])
    with pytest.raises(ValueError, match="need 0 <= k"):
        model.reconstruct(model.coordinates(DATA[1]), k)


def test_reconstruct_with_zero_components_is_the_mean():
    model = fit(DATA[1])
    rebuilt = model.reconstruct(model.coordinates(DATA[1]), 0)
    assert np.allclose(rebuilt, model.mean)


@pytest.mark.parametrize("d, k", [(5, 0), (0, 3), (5, -2)])
def test_random_projection_needs_positive_dimensions(d, k):
    with pytest.raises(ValueError, match="dimensions"):
        random_projection(d, k, rng)


def test_duplicate_rows_are_skipped_not_nan():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [4.0, 6.0]])
    ratios = squared_distance_ratios(X, np.eye(2))
    assert len(ratios) == 2 and np.allclose(ratios, 1)


def test_lda_works_when_a_column_copies_another():
    local = np.random.default_rng(5)
    a0, a1 = local.normal(0, 1, 200), local.normal(2, 1, 200)
    X0, X1 = np.column_stack([a0, a0]), np.column_stack([a1, a1])  # singular Σ
    w, c = lda(X0, X1)
    assert np.all(np.isfinite(w)) and accuracy(w, c, X0, X1) > 0.8


def test_lda_pinv_matches_solve_when_sigma_is_invertible():
    local = np.random.default_rng(6)
    X0 = local.normal(size=(100, 3))
    X1 = local.normal(size=(100, 3)) + [1, 0.5, 0]
    w, _ = lda(X0, X1)
    pooled = (100 * covariance(X0) + 100 * covariance(X1)) / 200
    assert np.allclose(w, np.linalg.solve(pooled, X1.mean(0) - X0.mean(0)))
