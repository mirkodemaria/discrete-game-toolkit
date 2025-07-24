# Discrete‑Game Identification Toolkit

This is a general‐purpose Python toolkit for analysing identification in **static discrete games of incomplete information**. It extends Aguirregabiria & Mira (2019) — *“Identification of Games of Incomplete Information with Multiple Equilibria and Unobserved Heterogeneity”* — to any number of players, binary actions, multiple exogenous states and multiple latent types (finite‑mixture components).

> **Paper context**  
> The mixture‑of‑types framework distinguishes three unobservables — private information, common payoff shifters and sunspots — and shows that when the number of actions exceeds the number of mixture components the model is non‑parametrically identified (Aguirregabiria & Mira 2019, *Quantitative Economics*, 10: 1659‑1701).

---

## Features

| Function | What it does |
|----------|--------------|
| `solve_equilibrium` | Iterates best‑responses under a logit link to obtain equilibrium CCPs for every state × type |
| `compute_joint_distribution` | Mixes over latent types to get the unconditional distribution of action profiles |
| `compute_scores_general` | Builds analytical score vectors ∂log Q/∂θ for mixture weights and CCPs |
| `build_hessian_general` | Assembles the block‑diagonal Hessian of the log‑likelihood |
| `build_Dc_hp_general` | Constructs Jacobian of equilibrium constraints w.r.t. mixture weights + CCPs |
| `ranks_and_condition_numbers` | Returns ranks and condition numbers of key matrices to check identification |

*Logit link implemented; probit easily added by supplying its inverse‑link derivative.*

---

## Installation

```bash
git clone https://github.com/<your‑handle>/discrete‑game‑identification.git
cd discrete‑game‑identification
pip install numpy
```

---

## Quick-Start Example

```python
import numpy as np
from discrete_game_identification import (
    solve_equilibrium, compute_joint_distribution,
    compute_scores_general, build_hessian_general,
    build_Dc_hp_general, ranks_and_condition_numbers,
)

# Dimensions
nplayers   = 2
num_z      = 4          # z ∈ {0, 1/3, 2/3, 1}
num_kappa  = 2          # κ ∈ {A, B}

# Payoff parameters α_j(z,k) and β_{j,i}(z,k)
alpha = np.zeros((nplayers, num_z, num_kappa))
beta  = np.zeros((nplayers, nplayers, num_z, num_kappa))

zvals = np.arange(num_z) / (num_z - 1)

# Player 1
alpha[0]        = -2.0 + 5.0 * zvals[:,None] + 1.0 * np.arange(num_kappa)
beta[0,1]       = -2.0                       # no z or κ effect
# Player 2
alpha[1]        = -3.0 + 4.0 * zvals[:,None] + 0.5 * np.arange(num_kappa)
beta[1,0]       = -3.0

# 70 % probability that κ = A in every state
nstates = num_z ** nplayers
h_matrix = 0.7 * np.ones((nstates, num_kappa))

# Equilibrium + diagnostics
p_eq   = solve_equilibrium(alpha, beta)
Q, A   = compute_joint_distribution(p_eq, h_matrix)
scores = compute_scores_general(p_eq, h_matrix, A)
H      = build_hessian_general(scores, Q)
Dc_hp  = build_Dc_hp_general(p_eq, beta)
ranks, conds = ranks_and_condition_numbers(H, Dc_hp)

print(ranks)
print(conds)
```

---

## File Overview

- discrete_game_identification.py  # core toolkit
- example_usage.py                 # replicates the 2×2×2 example
- README.md                        # this file
- LICENSE                          # MIT

---

## Extending the Toolkit

- add probit or other link functions → modify solve_equilibrium and derivatives
- allow non‑binary actions → generalise action enumeration and probability formulas
- incorporate payoff‑parameter Jacobian (Dc_pi) for structural estimation
- Monte‑Carlo engine for power studies of rank tests

Pull requests welcome!

---

## Citations

Aguirregabiria, V. & Mira, P. (2019). “Identification of Games of Incomplete Information with Multiple Equilibria and Unobserved Heterogeneity.” Quantitative Economics, 10 (4), 1659‑1701.
