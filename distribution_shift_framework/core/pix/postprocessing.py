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

"""Implementation of post(-augmentation) processing steps."""

from typing import Tuple

import chex
import jax


def mixup(images: chex.Array,
          labels: chex.Array,
          alpha: float = 1.,
          beta: float = 1.,
          rng: chex.PRNGKey = None) -> Tuple[chex.Array, chex.Array]:
  """Interpolating two images to create a new image.

  Source: https://arxiv.org/abs/1710.09412

  Args:
    images: Minibatch of images.
    labels: One-hot encoded labels for minibatch.
    alpha: Alpha parameter for the beta law which samples the interpolation
      weight.
    beta: Beta parameter for the beta law which samples the interpolation
      weight.
    rng: Random number generator state.

  Returns:
    Images resulting from the interpolation of pairs of images
    and their corresponding weighted labels.
  """
  assert labels.shape == 2, 'Labels need to represent one-hot encodings.'
  batch_size = images.shape[0]
  lmbda_rng, rng = jax.random.split(rng)
  lmbda = jax.random.beta(lmbda_rng, a=alpha, b=beta, shape=())
  idx = jax.random.permutation(rng, batch_size)

  images_a = images
  images_b = images[idx, :, :, :]
  images = lmbda * images_a + (1. - lmbda) * images_b[idx, :]
  labels = lmbda * labels + (1. - lmbda) * labels[idx, :]
  return images, labels
