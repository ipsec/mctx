# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for the Sampled MuZero wrapper (mctx._src.sampled)."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp

import mctx
from mctx._src import base
from mctx._src import sampled


_BATCH_SIZE = 2
_NUM_ACTIONS_K = 4      # K sampled actions per node
_ACTION_DIM = 3         # dimensionality of continuous actions
_NUM_CHANCE_OUTCOMES = 8 # codebook size for stochastic
_NUM_SIMULATIONS = 4


def _make_root_data(rng_key):
  """Creates synthetic root data for testing."""
  k1, k2, k3 = jax.random.split(rng_key, 3)
  root_embedding = jax.random.normal(k1, (_BATCH_SIZE, 16))
  root_sampled_actions = jax.random.normal(
      k2, (_BATCH_SIZE, _NUM_ACTIONS_K, _ACTION_DIM))
  root_prior_logits = jax.random.normal(
      k3, (_BATCH_SIZE, _NUM_ACTIONS_K))
  root_value = jnp.zeros(_BATCH_SIZE)
  return root_embedding, root_sampled_actions, root_prior_logits, root_value


def _dummy_continuous_recurrent_fn(params, rng_key, continuous_action, state):
  """A trivial continuous recurrent_fn for testing."""
  del params
  batch_size = state.shape[0]
  k1, k2 = jax.random.split(rng_key)

  # Generate next sampled actions for the child node.
  next_sampled = jax.random.normal(
      k1, (batch_size, _NUM_ACTIONS_K, _ACTION_DIM))
  logits = jax.random.normal(k2, (batch_size, _NUM_ACTIONS_K))

  out = sampled.ContinuousRecurrentFnOutput(
      reward=jnp.zeros(batch_size),
      discount=jnp.ones(batch_size),
      prior_logits=logits,
      value=jnp.zeros(batch_size),
      next_sampled_actions=next_sampled,
  )
  # Next real state: simple linear transform (just so it's not static).
  next_state = state + continuous_action.sum(axis=-1, keepdims=True)
  return out, next_state


def _dummy_continuous_decision_fn(params, rng_key, continuous_action, state):
  """A trivial continuous decision recurrent_fn for stochastic testing."""
  del params, rng_key
  batch_size = state.shape[0]
  chance_logits = jnp.zeros((batch_size, _NUM_CHANCE_OUTCOMES))
  afterstate_value = jnp.zeros(batch_size)
  out = base.DecisionRecurrentFnOutput(
      chance_logits=chance_logits,
      afterstate_value=afterstate_value,
  )
  # Afterstate is raw (not wrapped in SampledStateWrapper).
  afterstate = state + continuous_action.sum(axis=-1, keepdims=True)
  return out, afterstate


def _dummy_continuous_chance_fn(params, rng_key, chance_outcome, afterstate):
  """A trivial continuous chance recurrent_fn for stochastic testing."""
  del params, chance_outcome
  batch_size = afterstate.shape[0]
  k1, k2 = jax.random.split(rng_key)

  next_sampled = jax.random.normal(
      k1, (batch_size, _NUM_ACTIONS_K, _ACTION_DIM))
  action_logits = jax.random.normal(k2, (batch_size, _NUM_ACTIONS_K))

  out = sampled.ContinuousChanceRecurrentFnOutput(
      action_logits=action_logits,
      value=jnp.zeros(batch_size),
      reward=jnp.zeros(batch_size),
      discount=jnp.ones(batch_size),
      next_sampled_actions=next_sampled,
  )
  next_state = afterstate * 0.9  # simple transform
  return out, next_state


class SampledStateWrapperTest(chex.TestCase):
  """Tests for the SampledStateWrapper pytree compatibility."""

  def test_pytree_indexing(self):
    """SampledStateWrapper must be indexable via jax.tree.map."""
    rng = jax.random.PRNGKey(0)
    state = jax.random.normal(rng, (_BATCH_SIZE, 16))
    actions = jax.random.normal(rng, (_BATCH_SIZE, _NUM_ACTIONS_K, _ACTION_DIM))
    wrapper = sampled.SampledStateWrapper(
        real_state=state,
        sampled_actions=actions,
    )
    # Simulate the indexing that search.py does: tree.embeddings[batch, node].
    # With a single node dimension prepended:
    num_nodes = 3
    tiled = jax.tree.map(
        lambda x: jnp.stack([x] * num_nodes, axis=1), wrapper)
    # Index node 0 for all batches.
    batch_range = jnp.arange(_BATCH_SIZE)
    node_idx = jnp.zeros(_BATCH_SIZE, dtype=jnp.int32)
    indexed = jax.tree.map(lambda x: x[batch_range, node_idx], tiled)
    chex.assert_trees_all_close(indexed.real_state, state)
    chex.assert_trees_all_close(indexed.sampled_actions, actions)


class TauFactorTest(chex.TestCase):
  """Tests temperature correction factor."""

  def test_uniform_sampling(self):
    self.assertAlmostEqual(sampled._tau_factor(1.0), 0.0)

  def test_high_temperature(self):
    self.assertAlmostEqual(sampled._tau_factor(1e6), 1.0, places=5)

  def test_temperature_2(self):
    self.assertAlmostEqual(sampled._tau_factor(2.0), 0.5)


class MakeRecurrentFnTest(chex.TestCase):
  """Tests for the MuZero/Gumbel recurrent_fn wrapper."""

  def test_output_shapes(self):
    wrapped = sampled.make_sampled_recurrent_fn(
        _dummy_continuous_recurrent_fn, sampling_temperature=1.0)
    rng = jax.random.PRNGKey(42)
    state = sampled.SampledStateWrapper(
        real_state=jnp.zeros((_BATCH_SIZE, 16)),
        sampled_actions=jnp.ones(
            (_BATCH_SIZE, _NUM_ACTIONS_K, _ACTION_DIM)),
    )
    action_idx = jnp.zeros(_BATCH_SIZE, dtype=jnp.int32)
    out, next_state = wrapped(None, rng, action_idx, state)

    chex.assert_shape(out.reward, (_BATCH_SIZE,))
    chex.assert_shape(out.discount, (_BATCH_SIZE,))
    chex.assert_shape(out.prior_logits, (_BATCH_SIZE, _NUM_ACTIONS_K))
    chex.assert_shape(out.value, (_BATCH_SIZE,))
    chex.assert_shape(
        next_state.sampled_actions,
        (_BATCH_SIZE, _NUM_ACTIONS_K, _ACTION_DIM))

  def test_uniform_temperature_zeros_logits(self):
    """With tau=1 (uniform sampling), logits should be zeroed out."""
    wrapped = sampled.make_sampled_recurrent_fn(
        _dummy_continuous_recurrent_fn, sampling_temperature=1.0)
    rng = jax.random.PRNGKey(42)
    state = sampled.SampledStateWrapper(
        real_state=jnp.zeros((_BATCH_SIZE, 16)),
        sampled_actions=jnp.ones(
            (_BATCH_SIZE, _NUM_ACTIONS_K, _ACTION_DIM)),
    )
    action_idx = jnp.zeros(_BATCH_SIZE, dtype=jnp.int32)
    out, _ = wrapped(None, rng, action_idx, state)
    # tau_factor = 0 → all logits become 0
    chex.assert_trees_all_close(
        out.prior_logits,
        jnp.zeros((_BATCH_SIZE, _NUM_ACTIONS_K)))


class SampledMuzeroPolicyTest(chex.TestCase):
  """Integration test: run full search through the unified entry point."""

  @parameterized.parameters(
      (sampled.PolicyType.MUZERO, {}),
      (sampled.PolicyType.GUMBEL_MUZERO,
       {'max_num_considered_actions': 2}),
  )
  def test_muzero_and_gumbel(self, policy_type, extra_kwargs):
    rng = jax.random.PRNGKey(0)
    root_emb, root_actions, root_logits, root_val = _make_root_data(rng)

    output = sampled.sampled_muzero_policy(
        policy_type=policy_type,
        params=None,
        rng_key=rng,
        root_embedding=root_emb,
        root_sampled_actions=root_actions,
        root_prior_logits=root_logits,
        root_value=root_val,
        continuous_fns={'recurrent': _dummy_continuous_recurrent_fn},
        num_simulations=_NUM_SIMULATIONS,
        sampling_temperature=2.0,
        **extra_kwargs,
    )
    chex.assert_shape(output.action, (_BATCH_SIZE,))
    chex.assert_shape(output.action_weights, (_BATCH_SIZE, _NUM_ACTIONS_K))
    # Actions must be valid indices into K.
    self.assertTrue(jnp.all(output.action >= 0))
    self.assertTrue(jnp.all(output.action < _NUM_ACTIONS_K))

  def test_stochastic(self):
    rng = jax.random.PRNGKey(1)
    root_emb, root_actions, root_logits, root_val = _make_root_data(rng)

    output = sampled.sampled_muzero_policy(
        policy_type=sampled.PolicyType.STOCHASTIC_MUZERO,
        params=None,
        rng_key=rng,
        root_embedding=root_emb,
        root_sampled_actions=root_actions,
        root_prior_logits=root_logits,
        root_value=root_val,
        continuous_fns={
            'decision': _dummy_continuous_decision_fn,
            'chance': _dummy_continuous_chance_fn,
        },
        num_simulations=_NUM_SIMULATIONS,
        sampling_temperature=2.0,
    )
    chex.assert_shape(output.action, (_BATCH_SIZE,))
    # Stochastic MuZero internally pads to A+C then masks back to A
    chex.assert_shape(output.action_weights, (_BATCH_SIZE, _NUM_ACTIONS_K))

  def test_invalid_policy_type_raises(self):
    rng = jax.random.PRNGKey(0)
    root_emb, root_actions, root_logits, root_val = _make_root_data(rng)
    with self.assertRaises(ValueError):
      sampled.sampled_muzero_policy(
          policy_type="not_a_policy",
          params=None,
          rng_key=rng,
          root_embedding=root_emb,
          root_sampled_actions=root_actions,
          root_prior_logits=root_logits,
          root_value=root_val,
          continuous_fns={'recurrent': _dummy_continuous_recurrent_fn},
          num_simulations=_NUM_SIMULATIONS,
      )

  def test_missing_keys_raises(self):
    rng = jax.random.PRNGKey(0)
    root_emb, root_actions, root_logits, root_val = _make_root_data(rng)
    with self.assertRaises(ValueError):
      sampled.sampled_muzero_policy(
          policy_type=sampled.PolicyType.STOCHASTIC_MUZERO,
          params=None,
          rng_key=rng,
          root_embedding=root_emb,
          root_sampled_actions=root_actions,
          root_prior_logits=root_logits,
          root_value=root_val,
          continuous_fns={'recurrent': _dummy_continuous_recurrent_fn},
          num_simulations=_NUM_SIMULATIONS,
      )


class PublicAPITest(chex.TestCase):
  """Verify all symbols are exported from mctx."""

  def test_exports(self):
    self.assertTrue(hasattr(mctx, 'sampled_muzero_policy'))
    self.assertTrue(hasattr(mctx, 'PolicyType'))
    self.assertTrue(hasattr(mctx, 'SampledStateWrapper'))
    self.assertTrue(hasattr(mctx, 'ContinuousRecurrentFnOutput'))
    self.assertTrue(hasattr(mctx, 'ContinuousChanceRecurrentFnOutput'))
    self.assertTrue(hasattr(mctx, 'make_sampled_recurrent_fn'))
    self.assertTrue(hasattr(mctx, 'make_sampled_stochastic_fns'))


if __name__ == '__main__':
  absltest.main()
