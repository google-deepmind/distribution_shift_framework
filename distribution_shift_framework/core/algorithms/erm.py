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

"""Empirical risk minimization for minimizing loss."""
import abc
from typing import Tuple

import chex
from distribution_shift_framework.core.algorithms import base
from distribution_shift_framework.core.algorithms import losses
from distribution_shift_framework.core.datasets import data_utils
import jax
import jax.numpy as jnp


class ERM(base.LearningAlgorithm):
  """Computes the empirical risk."""

  def __init__(self,
               loss_fn: base.LossFn = losses.softmax_cross_entropy,
               name: str = 'empirical_risk'):
    super().__init__(loss_fn=loss_fn, name=name)

  def __call__(self,
               logits: chex.Array,
               targets: chex.Array,
               reduction: str = 'mean',
               **unused_kwargs) -> Tuple[data_utils.ScalarDict, chex.Array]:
    loss = self.loss_fn(logits, targets, reduction=reduction)
    return {'loss': loss}, logits


class AbstractMMD(base.LearningAlgorithm):
  """Base class for the CORAL and MMD algorithms."""

  def __init__(self,
               mmd_weight: float = 1.,
               loss_fn: base.LossFn = losses.softmax_cross_entropy,
               name: str = 'coral'):
    super().__init__(loss_fn=loss_fn, name=name)
    self.mmd_weight = mmd_weight

  @abc.abstractmethod
  def _mmd(self, x: chex.Array, x_mask: chex.Array, y: chex.Array,
           y_mask: chex.Array) -> chex.Array:
    """Computes the MMD between two sets of masked features.

    Args:
      x: The first set of features.
      x_mask: Which of the x features should be considered.
      y: The second set of features.
      y_mask: Which of the y features should be considered.

    Returns:
      A tuple of the mean and covariance.
    """
    pass

  def __call__(self,
               logits: chex.Array,
               targets: chex.Array,
               property_vs: chex.Array,
               reduction: str = 'mean'
               ) -> Tuple[data_utils.ScalarDict, chex.Array]:
    """Compute the MMD loss where the domains are given by the properties."""
    pnum = property_vs.shape[-1]
    if len(property_vs.shape) != 2:
      raise ValueError(
          f'Properties have an unexpected shape: {property_vs.shape}.')

    # For each label, compute the difference in domain shift against all the
    # others.
    mmd_loss = {'loss': 0}
    property_pairs = []
    for i, property_v1 in enumerate(range(pnum)):
      for property_v2 in range(i + 1, pnum):
        property_pairs += [(property_v1, property_v2)]

    def compute_pair_loss(mmd_loss, pair_vs):
      property_v1, property_v2 = pair_vs

      # One hot encoding.
      mask1 = jnp.argmax(property_vs, axis=-1)[..., None] == property_v1
      mask2 = jnp.argmax(targets, axis=-1)[..., None] == property_v2

      loss = jax.lax.cond(
          jnp.minimum(mask1.sum(), mask2.sum()) > 1,
          lambda a: self._mmd(*a),
          lambda _: jnp.zeros(()),
          operand=(logits, mask1, logits, mask2))

      t_mmd_loss = {'loss': loss}
      mmd_loss = jax.tree_map(jnp.add, mmd_loss, t_mmd_loss)
      return (mmd_loss, 0)

    mmd_loss, _ = jax.lax.scan(compute_pair_loss, mmd_loss,
                               jnp.array(property_pairs))

    erm = self.loss_fn(logits, targets, reduction=reduction)
    # How well are we estimating the labels?
    top1_acc = (jnp.argmax(logits, axis=-1) == jnp.argmax(targets,
                                                          axis=-1)).mean()

    loss = mmd_loss['loss'] / (pnum * (pnum - 1)) * self.mmd_weight + erm
    mmd_loss['loss'] = loss
    mmd_loss['erm'] = erm
    mmd_loss['top1_acc'] = top1_acc
    return mmd_loss, logits


class CORAL(AbstractMMD):
  """The CORAL algorithm.

  Computes the empirical risk and enforces that feature distributions match
  across distributions (by minimizing the maximum mean discrepancy).
  """

  def __init__(self,
               coral_weight: float = 1.,
               loss_fn: base.LossFn = losses.softmax_cross_entropy,
               name: str = 'coral'):
    super().__init__(loss_fn=loss_fn, name=name, mmd_weight=coral_weight)

  def _mmd(self, x: chex.Array, x_mask: chex.Array, y: chex.Array,
           y_mask: chex.Array) -> chex.Array:
    """Computes the MMD between two sets of masked features.

    Args:
      x: The first set of features.
      x_mask: Which of the x features should be considered.
      y: The second set of features.
      y_mask: Which of the y features should be considered.

    Returns:
      A tuple of the mean and covariance.
    """
    mean_x = (x * x_mask).sum(0, keepdims=True) / x_mask.sum()
    mean_y = (y * y_mask).sum(0, keepdims=True) / y_mask.sum()
    cent_x = (x - mean_x) * x_mask
    cent_y = (y - mean_y) * y_mask

    # Compute the covariances of the inputs.
    cova_x = cent_x.T.dot(cent_x) / (x_mask.sum() - 1)
    cova_y = cent_y.T.dot(cent_y) / (y_mask.sum() - 1)

    d_x = x_mask.sum()
    d_y = y_mask.sum()

    mean_mse = ((mean_x - mean_y)**2).mean()
    cov_mse = ((cova_x - cova_y)**2 / (4 * d_x * d_y)).mean()
    return mean_mse + cov_mse
