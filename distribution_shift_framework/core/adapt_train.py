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

"""Just train twice algorithm."""
import abc
from typing import Callable, Tuple

import chex
from distribution_shift_framework.core.datasets import data_utils
import jax.numpy as jnp
import optax

Learner = Tuple[Tuple[data_utils.ScalarDict, chex.Array], optax.OptState]
LearnerFN = Callable[..., Learner]


class Adapt(abc.ABC):
  """Encasuplates adapting parameters and state with auxiliary information.

  Given some initial set of parameters and the loss to be optimized, this
  set of classes is free to update the underlying parameters via adaptation
  based on difficulty of samples (e.g. JTT) or via EWA.
  """

  @abc.abstractmethod
  def update(self, params: optax.Params, state: optax.OptState,
             global_step: chex.Array):
    """Updates and returns the new parameters and state.

    Args:
      params: The parameters returned at this step.
      state: The state returned at this step.
      global_step: The training step.

    Returns:
      The updated params and state.
    """

  @abc.abstractmethod
  def __call__(self, fn: LearnerFN, params: optax.Params, state: optax.OptState,
               global_step: chex.Array, inputs: data_utils.Batch,
               rng: chex.PRNGKey) -> Tuple[data_utils.ScalarDict, chex.Array]:
    """Adapts the stored parameters according to the given information.

    Args:
      fn: The loss function.
      params: The parameters of the model at this step.
      state: The state of the model at this step.
      global_step: The step in the training pipeline.
      inputs: The inputs to the loss function.
      rng: The random key

    Returns:
      The scalars and logits which have been appropriately adapted.
    """


class JTT(Adapt):
  """Implementation of JTT algorithm."""

  def __init__(self, lmbda: float, num_steps_in_first_iter: int):
    """Implementation of JTT.

    This algorithm first trains for some number of steps on the full training
    set. After this first stage, the parameters at the end of this stage are
    used to select the most difficult samples (those that are misclassified)
    and penalize the loss more heavily for these examples.

    Args:
      lmbda: How much to upsample the misclassified examples.
      num_steps_in_first_iter: How long to train on full dataset before
        computing the error set and reweighting misclassified samples.
    """
    super().__init__()
    self.lmbda = lmbda
    self.num_steps_in_first_iter = num_steps_in_first_iter
    self.init_params = None
    self.init_state = None

  def update(self, params: optax.Params, state: optax.OptState,
             global_step: chex.Array):
    """See parent."""
    if global_step < self.num_steps_in_first_iter:
      self.init_params = params
      self.init_state = state
      return params, state

    return self.init_params, self.init_state

  def set(self, params: optax.Params, state: optax.OptState):
    self.init_params = params
    self.init_state = state

  def __call__(
      self, fn: LearnerFN, params: optax.Params, state: optax.OptState,
      old_params: optax.Params, old_state: optax.OptState,
      global_step: chex.Array, inputs: data_utils.Batch,
      rng: chex.PRNGKey) -> Learner:
    """See parent."""
    # Get the correct predictions with the params from the 1st training stage.
    (scalars, logits), g_state = fn(old_params, old_state, rng, inputs)
    predicted_label = jnp.argmax(logits, axis=-1)
    correct = jnp.equal(predicted_label, inputs['label']).astype(jnp.float32)

    # And now use this to reweight the current loss.
    (scalars, logits), g_state = fn(params, state, rng, inputs)
    new_loss = ((1 - correct) * scalars['loss'] * self.lmbda +
                correct * scalars['loss'])

    # And return the correct loss for the stage of training.
    in_first_stage = global_step < self.num_steps_in_first_iter
    scalars['1stiter_loss'] = scalars['loss'].mean()
    scalars['loss'] = (scalars['loss'] * in_first_stage + new_loss *
                       (1 - in_first_stage)).mean()
    return (scalars, logits), g_state
