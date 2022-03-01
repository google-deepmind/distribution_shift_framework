#!/usr/bin/python
#
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adaptation algorithms for modifying model parameters."""
import abc
from typing import Callable, Sequence

from absl import logging
import chex
from distribution_shift_framework.core.datasets import data_utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


def _broadcast(tensor1, tensor2):
  num_ones = len(tensor1.shape) - len(tensor2.shape)
  return tensor2.reshape(tensor2.shape + (1,) * num_ones)


def _bt_mult(tensor1, tensor2):
  tensor2 = _broadcast(tensor1, tensor2)
  return tensor1 * tensor2


def _get_mean(tensor):
  if len(tensor.shape) == 1:
    return jnp.mean(tensor, keepdims=True)
  else:
    return jnp.mean(tensor, axis=(0, 1), keepdims=True)


def _split_and_reshape(tree1, tree2):
  """Resize tree1 look like tree2 and return the resized tree and the modulo."""
  tree1_reshaped = jax.tree_map(
      lambda a, b: a[:np.prod(b.shape[0:2])].reshape(b.shape), tree1, tree2)
  tree1_modulo = jax.tree_map(lambda a, b: a[np.prod(b.shape[0:2]):], tree1,
                              tree2)
  return tree1_reshaped, tree1_modulo


class Adapt(abc.ABC):
  """Class to encapsulate an adaptation framework."""

  @abc.abstractmethod
  def __init__(self, init_params: optax.Params, init_state: optax.OptState,
               forward: Callable[..., chex.Array]):
    """Initializes the adaptation algorithm.

    This operates as follows. Given a number of examples, the model can update
    the parameters as it sees fit. Then, the updated parameters are run on an
    unseen test set.

    Args:
      init_params: The original parameters of the model.
      init_state: The original state of the model.
      forward: The forward call to the model.
    """

  @abc.abstractmethod
  def update(self, inputs: data_utils.Batch, property_label: chex.Array,
             rng: chex.PRNGKey, **kwargs):
    """Updates the parameters of the adaptation algorithm.

    Args:
      inputs: The batch to be input to the model.
      property_label: The properties of the image.
      rng: The random key.
      **kwargs: Keyword arguments specific to the forward function.
    """

  @abc.abstractmethod
  def run(self, fn: Callable[..., chex.Array], property_label: chex.Array,
          **fn_kwargs):
    """Runs the adaptation algorithm on a given function.

    Args:
      fn: The function we wish to apply the adapted parameters to.
      property_label: The property labels of the input values.
      **fn_kwargs: Additional kwargs to be input to the function fn.

    Returns:
      The result of fn using the adapted parameters according to the
        property_label value.
    """


class BNAdapt(Adapt):
  """Implements batch norm adaptation for a set of properties.

  Given a set of properties, and initial parameters/state, the batch
  normalization statistics are updated for each property value.
  """

  def __init__(self,
               init_params: optax.Params,
               init_state: optax.OptState,
               forward: Callable[..., chex.Array],
               n_properties: int,
               n: int = 10,
               N: int = 100):
    """See parent."""
    super().__init__(
        init_params=init_params, init_state=init_state, forward=forward)
    self.init_params = init_params
    self.init_state = init_state
    # Set the init state to 0. This will mean we always take the local stats.
    self.empty_state = self._reset_state(self.init_state)

    self.n_properties = n_properties
    self.forward_fn = forward
    self.adapted_state = {n: None for n in range(n_properties)}
    self.interpolated_states = None

    # Set up parameters that control the amount of adaptation.
    self.w_new = n
    self.w_old = N

    # Set up the cached dataset values.
    self._cached_dataset = [None] * self.n_properties

  def _reset_state(self, old_state, keys=('average', 'hidden', 'counter')):
    """Set the average of the BN parameters to 0."""
    state = hk.data_structures.to_mutable_dict(old_state)
    for k in state.keys():
      if 'batchnorm' in k and 'ema' in k:
        logging.info('Resetting %s in BNAdapt.', k)
        for state_key in keys:
          state[k][state_key] = jnp.zeros_like(state[k][state_key])
    state = hk.data_structures.to_haiku_dict(state)
    return state

  def _update_state(self, old_state, new_state, sz):
    """Update the state using the old and new running state."""
    if old_state is None:
      old_state = self._reset_state(self.init_state)

    new_state = hk.data_structures.to_mutable_dict(new_state)
    for k in new_state.keys():
      if 'batchnorm' in k and 'ema' in k:
        new_state_k = new_state[k]['average']
        old_counter = _broadcast(old_state[k]['average'],
                                 old_state[k]['counter'])
        new_state_k = new_state_k * sz
        old_state_k = old_state[k]['average'] * old_counter

        counter = jnp.maximum(old_counter + sz, 1)
        new_state[k]['average'] = (new_state_k + old_state_k) / counter
        new_state[k]['counter'] = counter.squeeze()
    new_state = hk.data_structures.to_haiku_dict(new_state)
    return new_state

  def _interpolate_state(self, old_state, new_state):
    """Update the state using the old and new running state."""
    if new_state is None:
      return old_state

    new_state = hk.data_structures.to_mutable_dict(new_state)
    new_ratio = self.w_new / (self.w_new + self.w_old)
    old_ratio = self.w_old / (self.w_new + self.w_old)
    for k in new_state.keys():
      if 'batchnorm' in k and 'ema' in k:
        new_state[k]['average'] = (
            new_state[k]['average'] * new_ratio +
            old_state[k]['average'] * old_ratio)
    new_state = hk.data_structures.to_haiku_dict(new_state)
    return new_state

  def update(self, inputs: data_utils.Batch, property_label: chex.Array,
             rng: chex.PRNGKey, **kwargs):
    """See parent."""
    # First, update cached data.
    for n in range(0, self.n_properties):
      mask = property_label == n
      masked_batch = jax.tree_map(lambda a: a[mask], inputs)  # pylint: disable=cell-var-from-loop
      if self._cached_dataset[n] is None:
        self._cached_dataset[n] = masked_batch
      else:
        self._cached_dataset[n] = jax.tree_map(lambda *a: jnp.concatenate(a),
                                               self._cached_dataset[n],
                                               masked_batch)

    # Then, if there are enough samples of a property, update the BN stats.
    for n in range(0, self.n_properties):
      # Update the adapted states with the output of the property labels.
      if (self._cached_dataset[n]['image'].shape[0] < np.prod(
          inputs['image'].shape[0:2])):
        continue

      # There are enough samples to do a forward pass.
      batch, mod_batch = _split_and_reshape(self._cached_dataset[n], inputs)
      _, state = self.forward_fn(self.init_params, self.empty_state, rng, batch,
                                 **kwargs)

      # Take the average over the cross replicas.
      state = jax.tree_map(_get_mean, state)
      self._update_state(
          self.adapted_state[n], state, sz=np.prod(batch['image'].shape[:2]))
      self._cached_dataset[n] = mod_batch

  def set_up_eval(self):
    self.interpolated_states = [
        self._interpolate_state(
            new_state=self.adapted_state[n], old_state=self.init_state)
        for n in range(self.n_properties)
    ]

  def run(self, fn: Callable[..., Sequence[chex.Array]],
          property_label: chex.Array, **fn_kwargs):
    """See parent."""
    # Get the results for the initial parameters and state.
    result = fn(self.init_params, self.init_state, **fn_kwargs)

    # Compute the results for each set of properties.
    for n in range(0, self.n_properties):
      mask = property_label == n
      if mask.sum() == 0:
        continue

      # And update the result.
      result_prop = fn(self.init_params, self.interpolated_states[n],
                       **fn_kwargs)
      result = [
          _bt_mult(r, (1 - mask)) + _bt_mult(r_prop, mask)
          for r, r_prop in zip(result, result_prop)
      ]

    return result
