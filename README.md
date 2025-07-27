# Discrete‑Game Identification Toolkit

This is a general‐purpose Python toolkit for analysing identification in **static discrete games of incomplete information**. It extends Aguirregabiria & Mira (2019) — *“Identification of Games of Incomplete Information with Multiple Equilibria and Unobserved Heterogeneity”* — to any number of players, binary actions, multiple exogenous states, and multiple latent types.

> **Paper context**  
> The mixture‑of‑types framework distinguishes three unobservables (private information, common payoff shifters and sunspots) and shows that when the number of actions exceeds the number of mixture components the model is non‑parametrically identified (Aguirregabiria & Mira 2019, *Quantitative Economics*, 10: 1659‑1701).
>
> **Why you might care**  
> In empirical IO we often specify a structural entry/exit or adoption game, simulate equilibria, then try to estimate pay‑off primitives. 
> Identification asks: *could two different parameter vectors generate exactly the same observable distributions?*  
> If the answer is “yes”, your model is not identified. This toolkit computes the Hessian of the log‑likelihood and the Jacobian of the equilibrium constraints, then checks the rank of the stacked matrix. Full rank of this matrix ⇒ local identification.

---

## Quick Install & Smoke‑Test

```bash
git clone  <this‑repo>
cd empirical-io-discrete-game-toolkit        # project root
python -m pip install -e .                   # editable install
python example_usage.py                      # reproduces 2×2×2 baseline from Aguirregabiria and Mira (2019)
```

Console excerpt:

```
Ranks:
  rank_hessian   : 48
  rank_hessian_h : 16
  rank_Dc_hp     : 48
  rank_J         : 64

Condition numbers:
  cond_H : 1.20e+06
  cond_J : 1.10e+07
```

In this example which replicates Aguirregabiria and Mira (2019), the model is locally identified because the rank of the stacked matrix (64) matches the dimension of the parameter vector (16 states x 2 latent types x 2 players = 64).

---

## What’s Inside

| Layer | File / module | Purpose |
|-------|---------------|---------|
| **Core library** | `src/discrete_game_identification/`<br>  `game_identification.py` | Solver for equilibrium CCPs, joint‑prob calculator, score/Hessian builders, Jacobian and rank diagnostics |
| **One‑shot example** | `example_usage.py` | 2‑player, 4‑state, 2‑type replication of Aguirregabiria & Mira (2019) |
| **Theory primer** | `primer.ipynb` | Simple and quick 10-point introduction to the model and identification tests |
| **Hands‑on notebooks** | `notebooks/01_simulate_game.py` → simulate & plot<br>`notebooks/02_identification_test.py` → run diagnostics<br>`notebooks/03_structural_estimation.py` → micro‑MLE demo | Tutorials with plots |
| **Visual helpers** | inline functions in the notebooks | CCP curves, heat‑maps, singular‑value plots |

---

## Minimal Code Example

```python
import numpy as np
from discrete_game_identification import (
    solve_equilibrium, compute_joint_distribution,
    compute_scores_general, build_hessian_general,
    build_Dc_hp_general, ranks_and_condition_numbers
)

# 2 players, 3 state values each, single latent type
z = np.linspace(0,1,3)
alpha = np.zeros((2,3,1)); beta = np.zeros((2,2,3,1))
alpha[0,:,0] = -1 + 4*z;  beta[0,1,:,0] = -3
alpha[1,:,0] = -1 + 3*z;  beta[1,0,:,0] = -2

p_eq = solve_equilibrium(alpha, beta)              # CCPs
Q,_  = compute_joint_distribution(p_eq, np.ones((9,1)))
scores = compute_scores_general(p_eq, np.ones((9,1)), _)
H     = build_hessian_general(scores, Q)
Dc    = build_Dc_hp_general(p_eq, beta)
print(ranks_and_condition_numbers(H, Dc)[0]['rank_J'])
```

---

## Learning Path

1. **Read the 10‑point primer** (`primer/primer.md` or `primer.ipynb`) – quick guide to pick up on the theory behind this model.
2. **Run Notebook 01** – see how CCPs respond to payoff changes and plot joint probabilities.
3. **Run Notebook 02** – compute Hessian & Jacobian; interpret rank and condition numbers.
4. **Run Notebook 03** – perform a one‑parameter MLE to illustrate identification vs estimation.

---

## Extending the toolkit

You could very easily fork the repository and extending the toolkit by adding:

* **More link functions** – add probit by replacing `logistic`/`qlogit` and derivatives.
* **Multinomial actions** – generalise `generate_action_combinations` and probability formulas.
* **Structural‐parameter Jacobian** – if your goal is to identify pay‑off primitives.

---

## References

* **Aguirregabiria, V. & Mira, P.** (2019) “Identification of games of incomplete information with multiple equilibria and unobserved heterogeneity.” *Quantitative Economics* 10 (4): 1659‑1701.  
* **Tamer, E.** (2003) “Incomplete Models of Strategic Interaction.” *Econometrica*.

---

## License

[MIT](./LICENSE) – free for academic or commercial use.
