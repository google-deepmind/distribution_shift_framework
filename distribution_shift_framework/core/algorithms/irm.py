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

"""Invariant risk minimization for minimizing loss."""
from typing import Tuple

import chex
from distribution_shift_framework.core.algorithms import base
from distribution_shift_framework.core.algorithms import losses
from distribution_shift_framework.core.datasets import data_utils
import haiku as hk
import jax.numpy as jnp


class IRM(base.LearningAlgorithm):
  """Computes the invariant risk.

  This learning algorithm is based on that of Arjovosky et al. Invariant Risk
  Minimization. https://arxiv.org/abs/1907.02893.

  It enforces that the optimal classifiers for representations with different
  properties are the same.
  """

  def __init__(self,
               lambda_penalty: float = 1.,
               loss_fn: base.LossFn = losses.softmax_cross_entropy,
               name: str = 'invariant_risk'):
    super().__init__(loss_fn=loss_fn, name=name)
    self.penalty_weight = lambda_penalty

  def _apply_loss(self, weights, logits, targets):
    return self.loss_fn(logits * weights, targets, reduction='mean')

  def __call__(self,
               logits: chex.Array,
               targets: chex.Array,
               property_vs: chex.Array,
               reduction: str = 'mean'
               ) -> Tuple[data_utils.ScalarDict, chex.Array]:
    assert len(targets.shape) == 2
    erm = 0
    penalty = 0

    # For each property, estimate the weights of an optimal classifier.
    for property_v in range(property_vs.shape[-1]):
      if len(property_vs.shape) == 2:
        # One hot encoding.
        mask = jnp.argmax(property_vs, axis=-1)[..., None] == property_v
        masked_logits = mask * logits
        masked_targets = mask * targets
      else:
        raise ValueError(
            f'Properties have an unexpected shape: {property_vs.shape}.')

      weights = jnp.ones((1,))

      # Compute empirical risk.
      erm += self._apply_loss(weights, masked_logits, masked_targets)

      # Compute penalty.
      grad_fn = hk.grad(self._apply_loss, argnums=0)
      grad_1 = grad_fn(weights, masked_logits[::2], masked_targets[::2])
      grad_2 = grad_fn(weights, masked_logits[1::2], masked_targets[1::2])
      penalty += (grad_1 * grad_2).sum()

    # How well are we estimating the labels?
    top1_acc = (jnp.argmax(logits, axis=-1) == jnp.argmax(targets,
                                                          axis=-1)).mean()

    return {
        'loss': erm + self.penalty_weight * penalty,
        'erm': erm,
        'penalty': penalty,
        'top1_acc': top1_acc
    }, logits
