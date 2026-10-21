# MLE → least squares

Maximum likelihood estimation built up from coin flips to linear regression. The code
shows that the "minimize squared error" rule behind least squares is just
maximum likelihood with Gaussian noise.

> Concept from BU DX601 (Math for Data Science), Week 7. All code here is my own,
> written after the lesson.

## Why it matters

Most loss functions in ML aren't arbitrary. Squared error comes from assuming Gaussian
noise, and cross-entropy comes from a Bernoulli likelihood. If you know the likelihood
behind a loss, you know what you're assuming about your data.

## What's inside (`mle.py`)

| Step | Idea | Checked by |
|------|------|-----------|
| Coin likelihood | A likelihood is a function of the **parameter** `p`, with the data held fixed | `p = 0.7` beats `p = 0.5` for 7/10 heads |
| Max vs arg-max | A grid search must return the best `p`, not `np.argmax`'s index | Grid search matches the closed form `heads / n` |
| Log trick | 200 probabilities of 0.01 multiply to exactly `0.0`; their log-sum is −921 | Underflow test |
| Gaussian MLE | μ̂ = mean, σ̂² = mean squared deviation (divide by **n**, not n − 1) | Matches `np.var`, and nudging either parameter lowers the likelihood |
| **MLE = least squares** | NLL = (n/2)·log(2πσ²) + SSE/(2σ²). Only the SSE term depends on `w` | Gradient descent on the NLL matches the normal equations and `np.linalg.lstsq` |
| Step size | On `(x−3)²`, lr 0.1 converges, 0.5 lands in one step, 1.0 bounces forever, 1.1 diverges | Parametrized test |

## Run it

```bash
pip install -r requirements-dev.txt
python mle-least-squares/mle.py
pytest mle-least-squares
```

```
Coin, 7 heads in 10 flips
  closed form p* = 0.7
  grid search p* = 0.700

200 probabilities of 0.01: product = 0.0, sum of logs = -921.0

Linear model with Gaussian noise, true w = [1.5, -2.0]
  least squares (normal eq.) w = [ 1.4603 -2.0316]
  MLE via gradient descent   w = [ 1.4603 -2.0316]

Gradient descent on f(x) = (x - 3)^2 from x = 0
  lr = 0.1  -> converges  x_20 = 2.965
  lr = 0.5  -> one-step   x_20 = 3
  lr = 1.0  -> oscillates x_20 = 0
  lr = 1.1  -> diverges   x_20 = -112
```

## Gotchas I hit

- `0 * log(0)` is `nan` in floating point, even though the math says the term is 0.
  The coin likelihood skips zero-count terms and returns `-inf` for impossible data.
- The noise variance σ² scales the loss but never moves the best `w`. That's why
  ordinary least squares can ignore it.
