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

"""Adversarial training of latent values."""
from typing import Optional, Sequence, Tuple

import chex
from distribution_shift_framework.core.algorithms import base
from distribution_shift_framework.core.algorithms import losses
from distribution_shift_framework.core.datasets import data_utils
import haiku as hk
import jax.numpy as jnp


class DANN(base.LearningAlgorithm):
  """Uses adversarial training to train a property agnostic representation.

  Based on the work of Ganin et al. Domain-Adversarial Training of Neural
  Networks. https://jmlr.org/papers/volume17/15-239/15-239.pdf.

  This learnign setup takes a set of logits, property values, and targets. It
  then enforces that the logits contain *no* information about the set of
  properties.
  """

  def __init__(self,
               loss_fn: base.LossFn = losses.softmax_cross_entropy,
               property_loss_fn: base.LossFn = losses.softmax_cross_entropy,
               mlp_output_sizes: Sequence[int] = (),
               name: str = 'DANN'):
    super().__init__(loss_fn=loss_fn, name=name)

    # Implicit assumptions in the code require classification.
    assert loss_fn == losses.softmax_cross_entropy
    assert property_loss_fn == losses.softmax_cross_entropy

    self.mlp_output_sizes = mlp_output_sizes
    self.property_loss_fn = property_loss_fn

  def __call__(self,
               logits: chex.Array,
               targets: chex.Array,
               property_vs: chex.Array,
               reduction: str = 'mean'
               ) -> Tuple[data_utils.ScalarDict, chex.Array]:
    ###################
    # Standard loss.
    ###################

    # Compute the regular loss function.
    erm = self.loss_fn(logits, targets, reduction=reduction)

    return {'loss': erm}, logits

  def adversary(self,
                logits: chex.Array,
                property_vs: chex.Array,
                reduction: str = 'mean',
                targets: Optional[chex.Array] = None) -> data_utils.ScalarDict:
    ###################
    # Adversarial loss.
    ###################
    adv_net = hk.nets.MLP(
        tuple(self.mlp_output_sizes) + (property_vs.shape[-1],))

    # Get logits for estimating the property.
    adv_logits = adv_net(logits)
    # Enforce that the representation encodes nothing about the property values.
    adv_loss = self.property_loss_fn(
        adv_logits, property_vs, reduction=reduction)
    # How well are we estimating the property value?
    prop_top1_acc = (jnp.argmax(adv_logits,
                                axis=-1) == jnp.argmax(property_vs,
                                                       axis=-1)).mean()

    return {'loss': adv_loss, 'prop_top1_acc': prop_top1_acc}
