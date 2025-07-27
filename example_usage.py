# Author: Mirko De Maria (July 2025)
# m.de-maria@imperial.ac.uk
# mirkodemaria.com

"""
Example usage of the discrete-game identification toolkit.

This script replicates the 2x2x2 example from Aguirregabiria and Mira
(2019) using the general toolkit provided in
``discrete_game_identification.py``. It constructs the payoff
parameters, solves for the equilibrium conditional choice
probabilities (CCPs), mixes over latent types to obtain the joint
distribution of actions, builds the Hessian and Jacobian matrices,
and computes ranks and condition numbers to assess identification.

Run this script from the command line::

    python example_usage.py

It should output the ranks of the Hessian, the restricted Hessian on
mixture weights, the Jacobian, and the stacked matrix, as well as
condition numbers of various matrices. The results should mirror
those obtained in the Gauss replication for the end-paper example.
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

import numpy as np
from discrete_game_identification import (
    solve_equilibrium,
    compute_joint_distribution,
    compute_scores_general,
    build_hessian_general,
    build_Dc_hp_general,
    ranks_and_condition_numbers,
)


def main() -> None:
    # Model dimensions
    nplayers = 2
    num_z = 4
    num_kappa = 2

    # Exogenous state values for each player (0, 1/3, 2/3, 1)
    zvals = np.arange(num_z) / (num_z - 1)

    # Payoff intercepts α_j(z_j,k) and slopes β_{j,i}(z_j,k)
    # Shapes: α[j,z,k], β[j,i,z,k]
    alpha = np.zeros((nplayers, num_z, num_kappa))
    beta = np.zeros((nplayers, nplayers, num_z, num_kappa))
    # Player 1 parameters
    a1_0, a1_z, a1_k = -2.0, 5.0, 1.0
    b1_0, b1_z, b1_k = -2.0, 0.0, 0.0
    alpha[0] = a1_0 + a1_z * zvals[:, None] + a1_k * np.arange(num_kappa)[None, :]
    beta[0, 1] = b1_0 + b1_z * zvals[:, None] + b1_k * np.arange(num_kappa)[None, :]
    # Player 2 parameters
    a2_0, a2_z, a2_k = -3.0, 4.0, 0.5
    b2_0, b2_z, b2_k = -3.0, 0.0, 0.0
    alpha[1] = a2_0 + a2_z * zvals[:, None] + a2_k * np.arange(num_kappa)[None, :]
    beta[1, 0] = b2_0 + b2_z * zvals[:, None] + b2_k * np.arange(num_kappa)[None, :]

    # Solve equilibrium CCPs for every state and type
    p_equil = solve_equilibrium(alpha, beta)

    # Mixture weights for each state and type (here constant 0.7 for κ=0)
    num_states = num_z ** nplayers
    h_matrix = 0.7 * np.ones((num_states, num_kappa))
    # Normalise mixture weights to sum to one across types
    h_matrix[:, 0] = 0.7
    h_matrix[:, 1] = 0.3

    # Compute joint distribution of actions
    Q, actions = compute_joint_distribution(p_equil, h_matrix)

    # Compute score vectors
    scores = compute_scores_general(p_equil, h_matrix, actions)

    # Build Hessian
    H = build_hessian_general(scores, Q)

    # Build Jacobian of equilibrium constraints
    Dc_hp = build_Dc_hp_general(p_equil, beta)

    # Compute ranks and condition numbers
    ranks, conds = ranks_and_condition_numbers(H, Dc_hp)

    # Display results
    print("Ranks:")
    for k, v in ranks.items():
        print(f"  {k}: {v}")
    print("\nCondition numbers:")
    for k, v in conds.items():
        print(f"  {k}: {v:.2e}")


if __name__ == '__main__':
    main()
