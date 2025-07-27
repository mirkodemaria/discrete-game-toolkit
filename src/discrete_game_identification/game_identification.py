"""
Discrete-Game Identification Toolkit
====================================

This module provides tools for analysing identification in static discrete
games of incomplete information. It generalises the 2x2x2 logistic
example of Aguirregabiria & Mira (2019) to models with an arbitrary
number of players, binary actions, multiple values of the players'
exogenous state variables, and multiple latent types (mixture
components). The toolkit computes equilibrium conditional choice
probabilities (CCPs) under a logit link, constructs the joint
distribution of actions by mixing over types, assembles score vectors
and the Hessian of the log-likelihood, and builds the Jacobian
matrices associated with the equilibrium constraints. It also
contains helper functions to check identification conditions via
matrix ranks and condition numbers.

Current limitations:

* The link function is logistic. Extension to other link
  functions (e.g., probit) requires a closed-form expression for the
  inverse link and its derivative.
* Payoff parameters enter linearly with a constant intercept and
  slopes on the probabilities of each other player. Each player's
  payoff for a given state and type has the form
  ``α_j(state, type) + Σ_{i≠j} β_{j,i}(state, type) * p_i``.
* The Jacobian with respect to payoff parameters (Dc_pi) is not
  provided here; users interested in identification of payoffs can
  adapt the pattern used in ``replicate_joint_identification.py``.

# Author: Mirko De Maria (July 2025)
# m.de-maria@imperial.ac.uk
# mirkodemaria.com
"""

from __future__ import annotations

import itertools
import math
from typing import List, Tuple, Iterable
import numpy as np


def logistic(x: np.ndarray) -> np.ndarray:
    """Logistic cumulative distribution function."""
    return 1.0 / (1.0 + np.exp(-x))


def qlogit(p: np.ndarray) -> np.ndarray:
    """Inverse logistic function."""
    return np.log(p) - np.log(1.0 - p)


def solve_equilibrium(
    alpha: np.ndarray,
    beta: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Solve for the equilibrium conditional choice probabilities for every
    combination of exogenous states and latent types.

    Parameters
    ----------
    alpha: Intercepts for each player, exogenous state and latent type.
    beta: Slope coefficients: beta[j,i,z,k] multiplies the CCP of player i
        in the payoff of player j. The diagonal should typically be
        zero.
    max_iter: Number of fixed-point iterations (default 1000). Optional.
    tol: Convergence tolerance on the L∞ norm of the CCPs (default 1e-12).

    Returns
    -------
    p_equil: Equilibrium CCPs for each state (ordered lexicographically by
        players' z indices), type and player.
    """
    nplayers, num_z, num_kappa = alpha.shape
    # Enumerate all state combinations: list of tuples (z1,z2,...)
    z_indices = list(itertools.product(range(num_z), repeat=nplayers))
    num_states = len(z_indices)
    p_equil = np.zeros((num_states, num_kappa, nplayers))
    # For each state and type, iterate until convergence
    for s, z_tuple in enumerate(z_indices):
        for k in range(num_kappa):
            p = np.zeros(nplayers)
            for _ in range(max_iter):
                # Compute utilities for each player
                util = np.zeros(nplayers)
                for j in range(nplayers):
                    z_j = z_tuple[j]
                    # Sum over other players' CCPs
                    interaction = 0.0
                    for i in range(nplayers):
                        if i != j:
                            z_i = z_tuple[i]
                            interaction += beta[j, i, z_j, k] * p[i]
                    util[j] = alpha[j, z_j, k] + interaction
                p_new = logistic(util)
                if np.max(np.abs(p_new - p)) < tol:
                    p = p_new
                    break
                p = p_new
            p_equil[s, k] = p
    return p_equil


def generate_action_combinations(nplayers: int) -> List[Tuple[int, ...]]:
    """Generate all possible action profiles for n players (binary actions)."""
    return list(itertools.product((0, 1), repeat=nplayers))


def compute_joint_distribution(
    p_equil: np.ndarray,
    h_matrix: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """
    Compute the joint distribution of actions Q(a|state) for each state by
    mixing over latent types.

    Parameters
    ----------
    p_equil: Equilibrium CCPs.
    h_matrix: Mixture weights for each state and type.  Each row should
        sum to 1.

    Returns
    -------
    Q: Unconditional probability of each action profile at each state.
    actions: list of tuples
        Action profiles corresponding to the columns of Q.
    """
    num_states, num_kappa, nplayers = p_equil.shape
    actions = generate_action_combinations(nplayers)
    num_actions = len(actions)
    Q = np.zeros((num_states, num_actions))
    for s in range(num_states):
        for a_idx, a in enumerate(actions):
            prob = 0.0
            for k in range(num_kappa):
                p = p_equil[s, k]
                # Probability of action a under type k
                prob_k = 1.0
                for j, a_j in enumerate(a):
                    prob_k *= p[j] if a_j == 1 else (1.0 - p[j])
                prob += h_matrix[s, k] * prob_k
            Q[s, a_idx] = prob
    return Q, actions


def compute_scores_general(
    p_equil: np.ndarray,
    h_matrix: np.ndarray,
    actions: Iterable[Tuple[int, ...]],
) -> List[np.ndarray]:
    """
    Compute score vectors ∂Q(a)/∂θ for every state and action profile.

    Parameters
    ----------
    p_equil: Equilibrium CCPs.
    h_matrix: Mixture weights for each state and type.
    actions: iterable of tuples
        Action profiles as returned by ``generate_action_combinations``.

    Returns
    -------
    scores: list of length num_states
        Each element is a 2D array of shape (num_actions, param_dim)
        containing the score vectors for all action profiles at a
        particular state. The parameter ordering within a state is
        ``[h(0),…,h(K-1), p(0,0),…,p(0,nplayers-1), p(1,0),…,p(K-1,nplayers-1)]``.
    """
    num_states, num_kappa, nplayers = p_equil.shape
    actions = list(actions)
    num_actions = len(actions)
    # Parameter dimension per state: mixture probs + mixture*players CCPs
    param_dim = num_kappa + num_kappa * nplayers
    scores = []
    for s in range(num_states):
        h = h_matrix[s]
        p_s = p_equil[s]
        state_scores = np.zeros((num_actions, param_dim))
        for a_idx, a in enumerate(actions):
            # Compute probabilities under each type
            prob_k = np.ones(num_kappa)
            for k in range(num_kappa):
                for j in range(nplayers):
                    prob_k[k] *= p_s[k, j] if a[j] == 1 else (1.0 - p_s[k, j])
            # Unconditional Q(a)
            Q_a = np.dot(h, prob_k)
            # Derivatives w.r.t. mixture weights
            state_scores[a_idx, :num_kappa] = prob_k
            # Derivatives w.r.t. p_j for each type k
            offset = num_kappa
            for k in range(num_kappa):
                for j in range(nplayers):
                    p_jk = p_s[k, j]
                    # derivative of prob_k w.r.t. p_jk
                    if a[j] == 1:
                        dprob = prob_k[k] / p_jk
                    else:
                        dprob = -prob_k[k] / (1.0 - p_jk)
                    # derivative of Q(a) w.r.t. p_jk
                    state_scores[a_idx, offset] = h[k] * dprob
                    offset += 1
        scores.append(state_scores)
    return scores


def build_hessian_general(
    scores: List[np.ndarray],
    Q: np.ndarray
) -> np.ndarray:
    """
    Assemble the block-diagonal Hessian of the log-likelihood for the
    general game. Each block corresponds to a state (combination of
    exogenous variables) and is computed as

        H_s = ∑_{a} (1/Q_s[a]) * score_s[a] * score_s[a]ᵀ

    Parameters
    ----------
    scores : list of score matrices
        Output of ``compute_scores_general``; one matrix per state.
    Q : Joint distribution of actions at each state.

    Returns
    -------
    hessian : Block-diagonal Hessian matrix.
    """
    num_states = len(scores)
    param_dim = scores[0].shape[1]
    hessian = np.zeros((num_states * param_dim, num_states * param_dim))
    for s, score_s in enumerate(scores):
        block = np.zeros((param_dim, param_dim))
        for a_idx in range(score_s.shape[0]):
            if Q[s, a_idx] > 0:
                v = score_s[a_idx]
                block += np.outer(v, v) / Q[s, a_idx]
        i = s * param_dim
        hessian[i:i+param_dim, i:i+param_dim] = block
    return hessian


def build_Dc_hp_general(
    p_equil: np.ndarray,
    beta: np.ndarray,
    link: str = 'logit',
) -> np.ndarray:
    """
    Construct the Jacobian of the equilibrium constraints with respect
    to the latent mixture weights and CCP parameters. Only the
    derivatives with respect to the CCPs are non-zero; derivatives
    w.r.t. mixture weights vanish because the constraints depend
    solely on the CCPs.

    For each state ``s``, latent type ``k`` and player ``j``, the
    equilibrium condition is::

        c_{j,k}(s) = α_{j,z_j,k} + Σ_{i≠j} β_{j,i,z_j,k} * p_{i,k}(s) - qlink(p_{j,k}(s)) = 0.

    The derivative with respect to p_{j,k}(s) is ``-qlink'(p_{j,k})``.
    The derivative with respect to p_{i,k}(s) for i ≠ j is ``β_{j,i,z_j,k}``.

    Parameters
    ----------
    p_equil: Equilibrium CCPs.
    beta: Coefficients used in the payoff function.
    link: Currently only ``'logit'`` is supported.

    Returns
    -------
    Dc_hp: Jacobian of the constraints with respect to parameters.  The
        columns are ordered as in the score vectors: mixture weights
        followed by CCPs.
    """
    num_states, num_kappa, nplayers = p_equil.shape
    num_z = beta.shape[2]
    # number of parameters per state
    param_dim = num_kappa + num_kappa * nplayers
    rows = num_states * nplayers * num_kappa
    cols = num_states * param_dim
    Dc_hp = np.zeros((rows, cols))
    for s in range(num_states):
        # decode z indices for this state
        # We'll reconstruct z_tuple from s using base‑num_z representation
        # to obtain z_j for each player
        z_tuple = []
        tmp = s
        for _ in range(nplayers):
            z_tuple.append(tmp % num_z)
            tmp //= num_z
        # row offset for state s
        row_base = s * nplayers * num_kappa
        col_base = s * param_dim
        for k in range(num_kappa):
            for j in range(nplayers):
                row_idx = row_base + k * nplayers + j
                # columns associated with CCPs start after mixture weights
                offset = col_base + num_kappa + k * nplayers
                for i in range(nplayers):
                    col_idx = offset + i
                    if i == j:
                        # derivative of qlink(p) w.r.t. p_jk
                        pjk = p_equil[s, k, j]
                        if link == 'logit':
                            dq = -1.0 / (pjk * (1.0 - pjk))
                        else:
                            raise NotImplementedError("Only logit link supported")
                        Dc_hp[row_idx, col_idx] = dq
                    else:
                        # derivative w.r.t p_{i,k}
                        z_j = z_tuple[j]
                        Dc_hp[row_idx, col_idx] = beta[j, i, z_j, k]
    return Dc_hp


def ranks_and_condition_numbers(
    hessian: np.ndarray,
    Dc_hp: np.ndarray,
    num_states: int | None = None,
    num_kappa: int | None = None,
    nplayers: int | None = None,
) -> Tuple[dict, dict]:
    """
    Compute ranks and condition numbers of key matrices used in
    identification analysis.

    This routine returns the rank of the full Hessian, the rank of
    the Hessian restricted to mixture-weight parameters, the rank of
    the Jacobian of the equilibrium constraints, and the rank of the
    stacked matrix ``[H; Dc_hp]``.  It also computes condition
    numbers for ``H'H``, ``H_h'H_h``, ``Dc_hp'Dc_hp`` and the
    combined matrix ``J = [H; Dc_hp]``.

    Parameters
    ----------
    hessian: Block-diagonal Hessian of the log-likelihood where ``m``
        denotes the number of parameters per state.
    Dc_hp: Jacobian of the equilibrium constraints with respect to the
        parameters.  Each state contributes ``nplayers * num_kappa``
        rows.
    num_states: 
        Number of distinct exogenous state configurations.  If not
        supplied, the function infers it by matching the row and
        column dimensions of ``hessian`` and ``Dc_hp``.
    num_kappa: 
        Number of latent types.  If omitted, it is inferred from the
        structure of ``Dc_hp`` under the assumption that each state
        contributes a block of size ``nplayers * num_kappa`` rows.
    nplayers: 
        Number of players.  If omitted, it is inferred along with
        ``num_kappa`` from the shape of ``Dc_hp``.

    Returns
    -------
    ranks:
        ``rank_hessian`` is the rank of the full Hessian; ``rank_hessian_h``
        is the rank of the Hessian restricted to mixture weights;
        ``rank_Dc_hp`` is the rank of ``Dc_hp``; and ``rank_J`` is the
        rank of the stacked matrix ``J = [hessian; Dc_hp]``.
    conds: 
        ``cond_H`` is the condition number of ``H'H``; ``cond_H_h`` is
        the condition number of ``H_h'H_h``; ``cond_Dc`` is the
        condition number of ``Dc_hp'Dc_hp``; ``cond_J`` is the
        condition number of ``J'J``.
    """
    # Infer model dimensions if not provided
    m = hessian.shape[0]  # total number of rows/cols of Hessian
    p = Dc_hp.shape[1]    # total number of parameters (columns)
    # Each block of the Hessian has dimension m_per_state = num_kappa*(1 + nplayers)
    if num_states is None or num_kappa is None or nplayers is None:
        # We know that Dc_hp has ns * nplayers * num_kappa rows
        # Let R = total rows; we need to factor R into integers
        R = Dc_hp.shape[0]
        # Let m_block = p / ns be param dimension per state (p = ns * m_block)
        # We find integers ns, nk, np such that:
        #   R = ns * np * nk
        #   p = ns * m_block
        #   m_block = nk * (1 + np)
        # Solve by trying divisors of R
        inferred = False
        for cand_ns in range(1, R + 1):
            if R % cand_ns == 0:
                # possible number of states
                rest = R // cand_ns
                # rest = nplayers * num_kappa
                # m_block = p / cand_ns
                if p % cand_ns != 0:
                    continue
                m_block = p // cand_ns
                # m_block must be divisible by something + maybe unify
                # Solve m_block = nk * (1 + np) and rest = np * nk
                # We try factors of rest
                for cand_nk in range(1, rest + 1):
                    if rest % cand_nk != 0:
                        continue
                    cand_np = rest // cand_nk
                    # Check if m_block matches
                    if m_block == cand_nk * (1 + cand_np):
                        num_states = cand_ns
                        num_kappa = cand_nk
                        nplayers = cand_np
                        inferred = True
                        break
                if inferred:
                    break
        if not inferred:
            raise ValueError(
                "Could not infer (num_states, num_kappa, nplayers) from matrix dimensions; please provide them explicitly."
            )

    # parameter dimension per state
    param_dim = num_kappa * (1 + nplayers)
    # Indices of mixture-weight parameters within each state's block
    mixture_cols = []
    mixture_rows = []
    for s in range(num_states):
        start = s * param_dim
        for k in range(num_kappa):
            mixture_cols.append(start + k)
            mixture_rows.append(start + k)
    # Extract submatrix H_h
    H_h = hessian[np.ix_(mixture_rows, mixture_cols)]
    # Build stacked matrix J
    J = np.vstack((hessian, Dc_hp))
    # Compute ranks
    ranks = {
        'rank_hessian': int(np.linalg.matrix_rank(hessian)),
        'rank_hessian_h': int(np.linalg.matrix_rank(H_h)),
        'rank_Dc_hp': int(np.linalg.matrix_rank(Dc_hp)),
        'rank_J': int(np.linalg.matrix_rank(J)),
    }
    # Helper function for condition number
    def cond(matrix: np.ndarray) -> float:
        """Return the condition number of a symmetric positive semi-definite matrix."""
        # Use singular values; ignore extremely small values to avoid underflow
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        s_nonzero = s[s > 1e-12]
        if s_nonzero.size == 0:
            return np.inf
        return s_nonzero.max() / s_nonzero.min()
    # Compute condition numbers
    conds = {
        'cond_H': cond(hessian.T @ hessian),
        'cond_H_h': cond(H_h.T @ H_h),
        'cond_Dc': cond(Dc_hp.T @ Dc_hp),
        'cond_J': cond(J.T @ J),
    }
    return ranks, conds
