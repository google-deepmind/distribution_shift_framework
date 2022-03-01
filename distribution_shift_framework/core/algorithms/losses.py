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

"""Common Losses to be used to train a model."""

import chex
import jax.numpy as jnp
import optax


def softmax_cross_entropy(logits: chex.Array,
                          labels: chex.Array,
                          reduction: str = 'sum') -> chex.Array:
  """Computes softmax cross entropy given logits and one-hot class labels.

  Args:
    logits: Logit output values.
    labels: Ground truth one-hot-encoded labels.
    reduction: Type of reduction to apply to loss.

  Returns:
    Loss value. If `reduction` is `none`, this has the same shape as `labels`;
    otherwise, it is scalar.

  Raises:
    ValueError: If the type of `reduction` is unsupported.
  """
  x = optax.softmax_cross_entropy(logits, labels)
  if reduction == 'none' or reduction is None:
    return jnp.asarray(x)
  elif reduction == 'sum':
    return jnp.asarray(x).sum()
  elif reduction == 'mean':
    return jnp.mean(jnp.asarray(x))
  else:
    raise ValueError('Unsupported reduction option.')
