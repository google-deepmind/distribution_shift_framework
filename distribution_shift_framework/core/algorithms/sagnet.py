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

"""Training representations to be style agnostic.."""
from typing import Any, Mapping, Optional, Tuple

import chex
from distribution_shift_framework.core.algorithms import base
from distribution_shift_framework.core.algorithms import losses
from distribution_shift_framework.core.datasets import data_utils
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections


class SagNet(base.LearningAlgorithm):
  """Implemenets a SagNet https://arxiv.org/pdf/1910.11645.pdf.

  This is a method for training networks to be invariant to style for
  improved domain generalization.
  """

  def __init__(self,
               loss_fn: base.LossFn = losses.softmax_cross_entropy,
               content_net_fn=hk.nets.MLP,
               content_net_kwargs: Mapping[str,
                                           Any] = (ml_collections.ConfigDict(
                                               dict(output_sizes=(64, 64,
                                                                  64)))),
               style_net_fn=hk.nets.MLP,
               style_net_kwargs: Mapping[str, Any] = ml_collections.ConfigDict(
                   dict(output_size=(64, 64, 64))),
               name: str = 'SagNet',
               **kwargs):
    super().__init__(loss_fn=loss_fn, name=name)
    self._content_net_fn = content_net_fn
    self._content_net_kwargs = content_net_kwargs

    self._style_net_fn = style_net_fn
    self._style_net_kwargs = style_net_kwargs

  def _randomize(self, features, interpolate=False, eps=1e-5):
    """Apply the ADAIN style operator (https://arxiv.org/abs/1703.06868)."""
    b = features.shape[0]
    alpha = jax.random.uniform(hk.next_rng_key(),
                               (b,) + (1,) * len(features.shape[1:]))

    is_image_shape = len(features.shape) == 4
    if is_image_shape:
      # Features is an image of with shape BHWC.
      b, h, w, c = features.shape
      features = jnp.transpose(features, axes=(0, 3, 1, 2)).view(b, c, -1)

    mean = jnp.mean(features, axis=(-1,), keepdims=True)
    variance = jnp.var(features, axis=(-1,), keepdims=True)
    features = (features - mean) / jnp.sqrt(variance + eps)

    idx_swap = jax.random.permutation(hk.next_rng_key(), jnp.arange(b))
    if interpolate:
      mean = alpha * mean + (1 - alpha) * mean[idx_swap, ...]
      variance = alpha * variance + (1 - alpha) * variance[idx_swap, ...]
    else:
      features = jax.lax.stop_gradient(features[idx_swap, ...])

    features = features * jnp.sqrt(variance + eps) + mean
    if is_image_shape:
      features = jnp.transpose(features, axes=(0, 2, 1)).view(b, h, w, c)
    return features

  def _content_pred(self, features):
    features = self._randomize(features, True)
    return self._content_net_fn(**self._content_net_kwargs)(features)

  def _style_pred(self, features):
    features = self._randomize(features, False)
    return self._style_net_fn(**self._style_net_kwargs)(features)

  def __call__(self,
               logits: chex.Array,
               targets: chex.Array,
               property_vs: chex.Array,
               reduction: str = 'mean'
               ) -> Tuple[data_utils.ScalarDict, chex.Array]:
    """Train the content network."""
    if len(logits.shape) == 4:
      logits = jnp.mean(logits, axis=(1, 2))
    preds = self._content_pred(logits)
    loss_content = self.loss_fn(preds, targets)

    # How well are we estimating the content?
    top1_acc = (jnp.argmax(preds, axis=-1) == jnp.argmax(targets,
                                                         axis=-1)).mean()
    return {'loss': loss_content, 'top1_acc': top1_acc}, preds

  def adversary(self,
                logits: chex.Array,
                property_vs: chex.Array,
                reduction: str = 'mean',
                targets: Optional[chex.Array] = None) -> data_utils.ScalarDict:
    """Train the adversary which aims to predict style."""
    if len(logits.shape) == 4:
      logits = jnp.mean(logits, axis=(1, 2))
    preds = self._style_pred(logits)
    loss_style = self.loss_fn(preds, targets)
    # How well are we estimating the style?
    top1_acc = (jnp.argmax(preds, axis=-1) == jnp.argmax(targets,
                                                         axis=-1)).mean()
    return {'loss': loss_style, 'style_top1acc': top1_acc}
