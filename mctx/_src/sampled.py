# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sampled MuZero: universal wrapper for continuous action spaces.

Implements the Sampled MuZero approach (Hubert et al., 2021) as a composable
layer on top of the existing mctx policies. The key insight is that MCTS
operates on discrete action indices 0..K-1 internally; this module bridges
continuous actions by:

  1. Sampling K continuous actions at each node.
  2. Storing them alongside the real state in a pytree-compatible wrapper.
  3. Mapping discrete indices back to continuous actions inside the recurrent_fn.
  4. Applying the importance-sampling temperature correction from the paper.

Supports all three mctx policies:
  - muzero_policy
  - gumbel_muzero_policy
  - stochastic_muzero_policy

Reference:
  Hubert et al., "Learning and Planning in Complex Action Spaces", 2021.
  https://arxiv.org/abs/2104.06303
"""

import enum
from typing import Any, Callable, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from mctx._src import base
from mctx._src import policies


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class PolicyType(enum.Enum):
  """Selects which mctx search policy to use."""
  MUZERO = "muzero"
  GUMBEL_MUZERO = "gumbel_muzero"
  STOCHASTIC_MUZERO = "stochastic_muzero"


@chex.dataclass(frozen=True)
class SampledStateWrapper:
  """Pytree-compatible wrapper storing the real state and sampled actions.

  Attributes:
    real_state: the actual environment/model state embedding. Shape ``[B, ...]``.
    sampled_actions: the K continuous actions sampled for this node.
      Shape ``[B, K, action_dim]``.
  """
  real_state: chex.ArrayTree
  sampled_actions: chex.Array  # [B, K, action_dim]


@chex.dataclass(frozen=True)
class ContinuousRecurrentFnOutput:
  """Output of a continuous-action recurrent_fn for Sampled MuZero.

  This extends the standard ``RecurrentFnOutput`` with an extra field
  ``next_sampled_actions`` that provides the K actions to consider at
  the next node.

  Attributes:
    reward: ``[B]`` approximate reward from the state-action transition.
    discount: ``[B]`` discount between the reward and the value.
    prior_logits: ``[B, K]`` logits over the K sampled actions.
    value: ``[B]`` approximate value after the transition.
    next_sampled_actions: ``[B, K, action_dim]`` new continuous actions
      sampled for the child node.
  """
  reward: chex.Array
  discount: chex.Array
  prior_logits: chex.Array
  value: chex.Array
  next_sampled_actions: chex.Array  # [B, K, action_dim]


@chex.dataclass(frozen=True)
class ContinuousChanceRecurrentFnOutput:
  """Output of a continuous-action chance_fn for Sampled Stochastic MuZero.

  Attributes:
    action_logits: ``[B, K]`` logits over the K sampled actions at the
      next decision node.
    value: ``[B]`` state value after the chance transition.
    reward: ``[B]`` reward produced by the chance transition.
    discount: ``[B]`` discount at the new state.
    next_sampled_actions: ``[B, K, action_dim]`` new continuous actions
      sampled for the next decision node.
  """
  action_logits: chex.Array   # [B, K]
  value: chex.Array           # [B]
  reward: chex.Array          # [B]
  discount: chex.Array        # [B]
  next_sampled_actions: chex.Array  # [B, K, action_dim]


# Type aliases for the user-provided continuous functions.
ContinuousRecurrentFn = Callable[
    [base.Params, chex.PRNGKey, chex.Array, base.RecurrentState],
    Tuple[ContinuousRecurrentFnOutput, base.RecurrentState]]

ContinuousDecisionRecurrentFn = Callable[
    [base.Params, chex.PRNGKey, chex.Array, base.RecurrentState],
    Tuple[base.DecisionRecurrentFnOutput, base.RecurrentState]]

ContinuousChanceRecurrentFn = Callable[
    [base.Params, chex.PRNGKey, chex.Array, base.RecurrentState],
    Tuple[ContinuousChanceRecurrentFnOutput, base.RecurrentState]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tau_factor(sampling_temperature: float) -> float:
  """Importance-sampling correction factor from Sampled MuZero.

  When sampling_temperature=1 (uniform sampling), tau_factor=0 so logits
  become uniform.  As temperature -> inf, tau_factor -> 1 and the original
  logits are preserved.
  """
  return 1.0 - (1.0 / sampling_temperature)


# ---------------------------------------------------------------------------
# Factory: MuZero / Gumbel MuZero
# ---------------------------------------------------------------------------

def make_sampled_recurrent_fn(
    continuous_recurrent_fn: ContinuousRecurrentFn,
    sampling_temperature: float,
) -> base.RecurrentFn:
  """Wraps a continuous-action recurrent_fn into a discrete mctx RecurrentFn.

  The returned function:
    1. Maps the discrete ``action_index`` to a continuous action via the
       ``sampled_actions`` stored in the state wrapper.
    2. Calls the user's continuous recurrent_fn with the real state.
    3. Applies the Sampled MuZero temperature correction to the prior logits.
    4. Packs the new sampled actions into the next state wrapper.

  Args:
    continuous_recurrent_fn: user function with signature
      ``(params, rng_key, continuous_action, real_state)``
      returning ``(ContinuousRecurrentFnOutput, next_real_state)``.
    sampling_temperature: temperature used when sampling the K actions.
      Controls the importance-sampling correction applied to the logits.

  Returns:
    A standard mctx ``RecurrentFn``.
  """
  tau = _tau_factor(sampling_temperature)

  def recurrent_fn(
      params: base.Params,
      rng_key: chex.PRNGKey,
      action_index: chex.Array,
      state: SampledStateWrapper,
  ) -> Tuple[base.RecurrentFnOutput, SampledStateWrapper]:
    # 1. Map discrete index -> continuous action
    batch_range = jnp.arange(action_index.shape[0])
    continuous_action = state.sampled_actions[batch_range, action_index]

    # 2. Call user's continuous function
    out, next_real_state = continuous_recurrent_fn(
        params, rng_key, continuous_action, state.real_state)

    # 3. Temperature correction (Sampled MuZero eq. 5)
    adjusted_logits = out.prior_logits * tau

    # 4. Pack next state
    next_state = SampledStateWrapper(
        real_state=next_real_state,
        sampled_actions=out.next_sampled_actions,
    )
    mctx_out = base.RecurrentFnOutput(
        reward=out.reward,
        discount=out.discount,
        prior_logits=adjusted_logits,
        value=out.value,
    )
    return mctx_out, next_state

  return recurrent_fn


# ---------------------------------------------------------------------------
# Factory: Stochastic MuZero
# ---------------------------------------------------------------------------

def make_sampled_stochastic_fns(
    continuous_decision_fn: ContinuousDecisionRecurrentFn,
    continuous_chance_fn: ContinuousChanceRecurrentFn,
    sampling_temperature: float,
) -> Tuple[base.DecisionRecurrentFn, base.ChanceRecurrentFn]:
  """Wraps continuous decision/chance functions for Stochastic MuZero.

  The decision_fn wrapper maps discrete indices to continuous actions.
  The chance_fn wrapper applies the temperature correction and packs
  the new sampled actions into the state.

  Args:
    continuous_decision_fn: ``(params, rng, continuous_action, real_state)``
      → ``(DecisionRecurrentFnOutput, afterstate_embedding)``.
    continuous_chance_fn: ``(params, rng, chance_outcome, afterstate)``
      → ``(ContinuousChanceRecurrentFnOutput, next_real_state)``.
    sampling_temperature: temperature for the importance-sampling correction.

  Returns:
    ``(decision_fn, chance_fn)`` tuple compatible with
    ``mctx.stochastic_muzero_policy``.
  """
  tau = _tau_factor(sampling_temperature)

  def decision_fn(
      params: base.Params,
      rng_key: chex.PRNGKey,
      action_index: chex.Array,
      state: SampledStateWrapper,
  ) -> Tuple[base.DecisionRecurrentFnOutput, base.RecurrentState]:
    # Map discrete index -> continuous action
    batch_range = jnp.arange(action_index.shape[0])
    continuous_action = state.sampled_actions[batch_range, action_index]
    # Delegate to user's function with the unwrapped real state.
    # Returns (DecisionRecurrentFnOutput, afterstate_embedding).
    return continuous_decision_fn(
        params, rng_key, continuous_action, state.real_state)

  def chance_fn(
      params: base.Params,
      rng_key: chex.PRNGKey,
      chance_outcome: chex.Array,
      afterstate: base.RecurrentState,
  ) -> Tuple[base.ChanceRecurrentFnOutput, SampledStateWrapper]:
    # The afterstate comes directly from decision_fn (raw, not wrapped).
    out, next_real_state = continuous_chance_fn(
        params, rng_key, chance_outcome, afterstate)

    # Temperature correction on the action logits for the next decision node.
    adjusted_logits = out.action_logits * tau

    # Pack new sampled actions into the state wrapper.
    next_state = SampledStateWrapper(
        real_state=next_real_state,
        sampled_actions=out.next_sampled_actions,
    )
    mctx_out = base.ChanceRecurrentFnOutput(
        action_logits=adjusted_logits,
        value=out.value,
        reward=out.reward,
        discount=out.discount,
    )
    return mctx_out, next_state

  return decision_fn, chance_fn


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_POLICY_DISPATCH = {
    PolicyType.MUZERO: policies.muzero_policy,
    PolicyType.GUMBEL_MUZERO: policies.gumbel_muzero_policy,
    PolicyType.STOCHASTIC_MUZERO: policies.stochastic_muzero_policy,
}


def sampled_muzero_policy(
    policy_type: PolicyType,
    params: base.Params,
    rng_key: chex.PRNGKey,
    root_embedding: chex.ArrayTree,
    root_sampled_actions: chex.Array,
    root_prior_logits: chex.Array,
    root_value: chex.Array,
    continuous_fns: Dict[str, Callable],
    num_simulations: int,
    sampling_temperature: float = 1.0,
    **kwargs: Any,
) -> base.PolicyOutput:
  """Run any mctx MCTS policy over a sampled continuous action space.

  This is the main entry point for Sampled MuZero.  It:
    1. Applies the importance-sampling temperature correction to root logits.
    2. Wraps the root state with the sampled actions.
    3. Creates the appropriate discrete recurrent_fn wrapper(s).
    4. Delegates to the selected mctx policy.

  Args:
    policy_type: which mctx policy to use (see ``PolicyType`` enum).
    params: parameters forwarded to the recurrent functions.
    rng_key: random number generator state (consumed).
    root_embedding: the real state embedding at the root. Shape ``[B, ...]``.
    root_sampled_actions: K continuous actions sampled at the root.
      Shape ``[B, K, action_dim]``.
    root_prior_logits: policy logits over the K sampled actions.
      Shape ``[B, K]``.
    root_value: estimated value at the root. Shape ``[B]``.
    continuous_fns: dictionary with the user's continuous functions.
      - For ``MUZERO`` / ``GUMBEL_MUZERO``:
        ``{'recurrent': ContinuousRecurrentFn}``
      - For ``STOCHASTIC_MUZERO``:
        ``{'decision': ContinuousDecisionRecurrentFn,
         'chance': ContinuousChanceRecurrentFn}``
    num_simulations: number of MCTS simulations to run.
    sampling_temperature: temperature used when sampling the K actions.
      Controls the importance-sampling correction (default 1.0 = uniform).
    **kwargs: extra keyword arguments forwarded to the underlying mctx policy
      (e.g. ``max_num_considered_actions`` for Gumbel MuZero,
      ``dirichlet_alpha``, ``temperature``, etc.).

  Returns:
    ``PolicyOutput`` with the selected action index (into the K sampled
    actions), action weights, and the search tree.

  Raises:
    ValueError: if ``policy_type`` is unknown or ``continuous_fns`` is missing
      required keys.
  """
  if policy_type not in _POLICY_DISPATCH:
    raise ValueError(
        f"Unknown policy_type={policy_type!r}. "
        f"Expected one of {list(PolicyType)}.")

  # 1. Temperature correction on root logits
  tau = _tau_factor(sampling_temperature)
  adjusted_root_logits = root_prior_logits * tau

  # 2. Wrap the root state
  sampled_root_state = SampledStateWrapper(
      real_state=root_embedding,
      sampled_actions=root_sampled_actions,
  )

  root = base.RootFnOutput(
      prior_logits=adjusted_root_logits,
      value=root_value,
      embedding=sampled_root_state,
  )

  # 3. Route to the correct policy
  if policy_type is PolicyType.STOCHASTIC_MUZERO:
    _check_keys(continuous_fns, ['decision', 'chance'], policy_type)
    decision_fn, chance_fn = make_sampled_stochastic_fns(
        continuous_decision_fn=continuous_fns['decision'],
        continuous_chance_fn=continuous_fns['chance'],
        sampling_temperature=sampling_temperature,
    )
    return policies.stochastic_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        decision_recurrent_fn=decision_fn,
        chance_recurrent_fn=chance_fn,
        num_simulations=num_simulations,
        **kwargs,
    )
  else:
    _check_keys(continuous_fns, ['recurrent'], policy_type)
    recurrent_fn = make_sampled_recurrent_fn(
        continuous_recurrent_fn=continuous_fns['recurrent'],
        sampling_temperature=sampling_temperature,
    )
    policy_fn = _POLICY_DISPATCH[policy_type]
    return policy_fn(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        **kwargs,
    )


def _check_keys(
    fns: Dict[str, Callable],
    required: list,
    policy_type: PolicyType,
) -> None:
  missing = [k for k in required if k not in fns]
  if missing:
    raise ValueError(
        f"continuous_fns is missing keys {missing} "
        f"required for {policy_type.value}.")
