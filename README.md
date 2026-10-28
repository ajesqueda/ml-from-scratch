# ml-from-scratch

Machine learning and the math underneath it, implemented from scratch in Python
and checked against NumPy and closed-form answers. The goal is to understand
*why* a method works, not just call a library.

Each folder is self-contained, with its own README, code and tests.

## Index

| Topic | What it shows | Source |
|-------|---------------|--------|
| [MLE → least squares](mle-least-squares/) | Likelihood, the log trick, Gaussian MLE, why least squares *is* MLE, gradient-descent step sizes | DX601 Week 7 |
| [Vector similarity](vector-similarity/) | L1/L2/max norms, Euclidean vs cosine, nearest neighbors in pure Python and NumPy, why feature scaling matters | DX601 Week 8 |
| [Gaussian elimination](gaussian-elimination/) | Row reduction with exact fractions and a step trace; 0/1/∞ solutions, parametric form, inverse via `[A \| I]`; messy equations → `Ax = b`; row vs column view of `Ax` | DX601 Week 9 |

## Run it

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

## Workflow

Work happens on `feat/*` branches off `dev`, then is promoted `dev` → `qa` → `main`
through pull requests. CI runs ruff and pytest on every PR.
