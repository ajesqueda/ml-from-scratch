"""Maximum likelihood estimation from scratch, ending at least squares.

The thread through this file:

1. A likelihood scores *parameters* given fixed data. MLE picks the best one.
2. Products of many probabilities underflow, so we maximize the log-likelihood.
3. For a coin, the MLE has a closed form (heads / n) that a grid search agrees with.
4. For Gaussian data, the MLE is the sample mean and the *mean* squared deviation.
5. For a linear model with Gaussian noise, maximizing likelihood is exactly
   minimizing squared error, so MLE re-derives least squares.
6. When there is no closed form, gradient descent walks downhill, and the step
   size decides whether it crawls, lands, bounces or blows up.

Run `python mle.py` for a short tour.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

# --- 1-3. Coin flips -------------------------------------------------------


def coin_log_likelihood(p: float, heads: int, n: int) -> float:
    """log Pr[data | p] for `heads` out of `n` independent flips.

    The binomial coefficient is left out: it doesn't depend on p, so it can't
    change *which* p is best, only the height of the curve.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    tails = n - heads
    # Seeing a head when p = 0 (or a tail when p = 1) is impossible: log(0).
    if (heads and p == 0.0) or (tails and p == 1.0):
        return -math.inf
    # 0 * log(0) is mathematically 0 here, but numerically it's nan, so skip
    # a term entirely when its count is zero.
    log_heads = heads * math.log(p) if heads else 0.0
    log_tails = tails * math.log(1 - p) if tails else 0.0
    return log_heads + log_tails


def coin_mle_closed_form(heads: int, n: int) -> float:
    # Setting d/dp [h log p + t log(1-p)] = h/p - t/(1-p) to 0 gives p = h / n.
    return heads / n


def coin_mle_grid(heads: int, n: int, resolution: int = 1001) -> float:
    """Brute force: score every candidate p and keep the arg-max."""
    grid = np.linspace(0.0, 1.0, resolution)
    scores = [coin_log_likelihood(float(p), heads, n) for p in grid]
    # np.argmax returns an *index*, not the best p. Converting back is the
    # classic max vs arg-max trap.
    return float(grid[int(np.argmax(scores))])


def product_vs_log_sum(probabilities: Sequence[float]) -> tuple[float, float]:
    """Return (plain product, sum of logs) to show why logs are used.

    Multiplying a few hundred values around 0.01 rounds to exactly 0.0 in
    float64, which wipes out any comparison between models. The sum of logs
    stays an ordinary negative number.
    """
    product = 1.0
    for p in probabilities:
        product *= p
    return product, sum(math.log(p) for p in probabilities)


# --- 4. Gaussian ----------------------------------------------------------


def gaussian_log_likelihood(x: np.ndarray, mu: float, sigma2: float) -> float:
    n = x.size
    return float(-0.5 * n * math.log(2 * math.pi * sigma2) - ((x - mu) ** 2).sum() / (2 * sigma2))


def gaussian_mle(x: np.ndarray) -> tuple[float, float]:
    """MLE of (mean, variance). Note the variance divides by n, not n - 1."""
    mu = float(x.mean())
    return mu, float(((x - mu) ** 2).mean())


# --- 5. Linear regression: MLE == least squares ---------------------------


def gaussian_regression_nll(w: np.ndarray, X: np.ndarray, y: np.ndarray, sigma2: float) -> float:
    """Negative log-likelihood of y = Xw + Gaussian noise.

    Expand it and you get  (n/2) log(2 pi sigma2) + SSE(w) / (2 sigma2).
    The first term doesn't involve w, and the second is the sum of squared
    errors scaled by a positive constant, so the w that minimizes the NLL is
    the w that minimizes squared error.
    """
    residuals = y - X @ w
    n = y.size
    return float(0.5 * n * math.log(2 * math.pi * sigma2) + (residuals @ residuals) / (2 * sigma2))


def least_squares_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Solve (X^T X) w = X^T y. Using solve() is more stable than inverting.
    return np.linalg.solve(X.T @ X, X.T @ y)


def fit_regression_by_mle(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 5_000
) -> np.ndarray:
    """Minimize the Gaussian NLL with gradient descent (sigma2 fixed at 1).

    The gradient of SSE/2 is -X^T (y - Xw). Dividing by n keeps the step size
    independent of how many rows there are.
    """
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= lr * (-(X.T @ (y - X @ w)) / y.size)
    return w


# --- 6. Gradient descent on a 1-D bowl ------------------------------------


def gradient_descent_1d(
    grad: Callable[[float], float], x0: float, lr: float, steps: int
) -> list[float]:
    """Return every x visited, so the path itself can be inspected."""
    path = [x0]
    for _ in range(steps):
        path.append(path[-1] - lr * grad(path[-1]))
    return path


def classify_step_size(path: Sequence[float], target: float, tol: float = 1e-6) -> str:
    """Label a descent path as 'one-step', 'converges', 'oscillates' or 'diverges'.

    Judged by the distance to the target: shrinking means converging (maybe
    slowly), growing means diverging, and staying the same means bouncing
    back and forth across the minimum forever.
    """
    errors = [abs(x - target) for x in path]
    if errors[1] < tol:
        return "one-step"
    if errors[-1] > errors[0] + tol:
        return "diverges"
    if errors[-1] < errors[0] - tol:
        return "converges"
    return "oscillates"


def main() -> None:
    print("Coin, 7 heads in 10 flips")
    print(f"  closed form p* = {coin_mle_closed_form(7, 10)}")
    print(f"  grid search p* = {coin_mle_grid(7, 10):.3f}")

    product, log_sum = product_vs_log_sum([0.01] * 200)
    print(f"\n200 probabilities of 0.01: product = {product}, sum of logs = {log_sum:.1f}")

    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.uniform(-2, 2, 200)])
    y = X @ np.array([1.5, -2.0]) + rng.normal(0, 0.5, 200)
    print("\nLinear model with Gaussian noise, true w = [1.5, -2.0]")
    print(f"  least squares (normal eq.) w = {least_squares_normal_equations(X, y).round(4)}")
    print(f"  MLE via gradient descent   w = {fit_regression_by_mle(X, y).round(4)}")

    print("\nGradient descent on f(x) = (x - 3)^2 from x = 0")
    for lr in (0.1, 0.5, 1.0, 1.1):
        path = gradient_descent_1d(lambda x: 2 * (x - 3), 0.0, lr, 20)
        print(f"  lr = {lr:<4} -> {classify_step_size(path, 3.0):<10} x_20 = {path[-1]:.4g}")


if __name__ == "__main__":
    main()
