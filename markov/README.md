# Markov chains and matrix powers

A Markov chain is a matrix power in disguise: tomorrow's probabilities are `P` times
today's, so the probabilities k days out are `Pᵏ` times today's. Eigenvectors explain
both what the chain settles into (the eigenvector for λ = 1) and how fast it forgets
where it started (the next-biggest |λ|). This folder computes steady states and
distributions exactly with fractions (the forgetting speed is a float estimate), building on [`eigen/`](../eigen/) and the exact null space in
[`vector-spaces/`](../vector-spaces/).

> Concept from BU DX601 (Math for Data Science), Week 12: matrix powers, power iteration and Markov chains.

## What's inside (`chains.py`)

| Function | What it does |
|----------|--------------|
| `steady_state(P)` | Solves `(P − I)s = 0` and scales s to add up to 1; swap-and-flip for 2×2, exact null space for bigger chains |
| `distribution_after(P, s0, k)` | `Pᵏ s0` by multiplying k times |
| `mixing_factor(P)` | \|λ₂\| (float): how fast the chain forgets where it started |
| `power_via_eigen(A, y, k)` | `Aᵏ y` from eigen-pieces, no repeated multiplying |
| `check_stochastic(P)` | Columns must be non-negative and sum to 1 (columns are "from") |
| `swap_and_flip(a, b)` | A non-zero solution of `a·s₁ + b·s₂ = 0` |

## Run it

```bash
pip install -r requirements-dev.txt
python markov/chains.py
pytest markov
```

```
P = [['9/10', '1/2'], ['1/10', '1/2']]
  P − I row 1: ['-1/10', '1/2'] -> swap and flip: ['1/2', '1/10']
  steady state: (5/6, 1/6)  check P s = s: True
  mixing factor |λ2| = 0.40
  start rainy, after  1 days: [0.5, 0.5]
  start rainy, after  2 days: [0.7, 0.3]
  start rainy, after  5 days: [0.8248, 0.1752]
  start rainy, after 10 days: [0.8332, 0.1668]
  start rainy, after 20 days: [0.8333, 0.1667]

A^k y via eigen-pieces for A = [[4, 1], [2, 3]] y = [7, 1]
  k=1: (29, 17)  direct: [29, 17]
  k=3: (641, 593)  direct: [641, 593]
  k=6: (78253, 77869)  direct: [78253, 77869]
```

## The steady state, step by step

The steady state is a distribution that one more step doesn't change: `P s = s`, which
is the eigenvector of P for λ = 1. Every Markov matrix has λ = 1, because its columns
add up to 1.

1. **Build `P − I`** (subtract 1 from the diagonal only).
2. **One equation or two?** For a 2×2, the rows of `P − I` are always multiples of each
   other, so there's really **one equation** `a·s₁ + b·s₂ = 0`. That's a free-variable
   situation: swap the two numbers and flip one sign, `s = (b, −a)`, and you're done.
   For the weather chain, row 1 is `(−1/10, 1/2)`, so `s = (1/2, 1/10)`.
3. **Scale to probabilities:** add **all** the entries (1/2 + 1/10 = 3/5) and divide
   **each** entry by that sum, keeping the order: `(5/6, 1/6)`. It must add to 1.
4. **Check:** `P s = s`. The tests check this on random 2×2 to 5×5 chains.

Some chains don't have one answer, and `steady_state` says so instead of returning
something misleading:

- `P = I`: nothing ever moves, so every distribution is steady.
- A chain that splits into separate groups of states has one steady state per group.
- The flip-flop chain `[[0, 1], [1, 0]]` has a steady state (½, ½), but starting from
  state 1 it alternates forever and never settles there. Its λ₂ = −1, and the next
  section explains why that matters.

## How fast a chain forgets where it started

Split the starting distribution into eigen-pieces. The λ = 1 piece is the steady state
and never changes. Every other piece shrinks by its |λ| on every step, so the slowest
one, **|λ₂|**, decides how long the start is remembered. For the weather chain
|λ₂| = 0.4: the gap to the steady state shrinks to 40% of itself each day, which is
why day 10 is already at 0.8332. When |λ₂| = 1 (flip-flop, or a split chain), the start
is never forgotten.

That "exactly 40% each day" holds for every 2-state chain. In bigger chains |λ₂| is the
**long-run** rate: if a repeated eigenvalue is missing an eigenvector, the gap picks up
an extra factor like k·|λ₂|ᵏ, so the first few steps can shrink by less. `mixing_factor`
is computed with floating-point eigenvalues, so it's an estimate, unlike the exact
steady states.

This is the power-iteration "shrink factor" question: |smaller λ| ÷ |bigger λ|, with no
multiplying needed. For a Markov chain the bigger λ is 1, so it's just |λ₂|.

## Aᵏy from eigen-pieces: two equations, so substitute into the other one

To compute `Aᵏ y` without multiplying k times, write y as a mix of the eigenvectors,
`y = c₁v₁ + c₂v₂`. Unlike the eigenvector step, these are **two different equations**
(one per entry of y), so it's substitution, not a free variable:

```
A = [[4, 1], [2, 3]]:  v₁ = (1, 1) for λ₁ = 5,  v₂ = (1, −2) for λ₂ = 2,  y = (7, 1)

entry 1:  c₁ + c₂ = 7     ->  c₁ = 7 − c₂
entry 2:  c₁ − 2c₂ = 1    ->  substitute into THIS one, not back into entry 1:
          (7 − c₂) − 2c₂ = 1  ->  c₂ = 2,  then c₁ = 7 − 2 = 5
```

Plugging `c₁ = 7 − c₂` back into entry 1 only gives `7 = 7`, which tells you nothing.
Then each piece just picks up its own λ every step:

```
Aᵏ y = 5·5ᵏ·(1, 1) + 2·2ᵏ·(1, −2)        k = 3:  (625 + 16, 625 − 32) = (641, 593)
```

Keep the other variable in the line while solving (`c₁ = 7 − c₂`), and write the full
vector at the end. `power_via_eigen` does exactly this, and the tests compare it with
`matrix_power` for k = 0 to 9, including a matrix with a negative eigenvalue.

## Changing the matrix changes λ simply

The eigenvectors stay the same, and the eigenvalues follow along:

| Matrix | Eigenvalues | Why |
|--------|-------------|-----|
| A² | λ² | A(Av) = A(λv) = λ²v |
| A + cI | λ + c | (A + cI)v = λv + cv |
| cA | cλ | (cA)v = c(λv) |
| A⁻¹ | 1/λ | apply A⁻¹ to Av = λv and divide by λ |

To check A² without solving its characteristic polynomial: its trace should equal the
sum of the λ², and its determinant their product. The tests check all of these on
random 3×3 matrices.

## Tests

`tests/test_chains.py` (79 tests): exact steady states for hand examples and decimal
inputs; `P s = s`, `sum(s) = 1` and positive entries on random 2×2 to 5×5 chains,
matching NumPy's λ = 1 eigenvector; convergence from any start; swap-and-flip and
normalizing; the identity, split and flip-flop chains; rows-vs-columns and negative
entries rejected; the gap to the steady state
shrinking by exactly |λ₂| per step (2-state chains);
`Aᵏy` from eigen-pieces matching `matrix_power`; and the eigenvalue rules with the
trace/det check; plus bad input: empty matrices, negative steps, a start
vector of the wrong length, a one-state chain, and floats read as their decimals.
