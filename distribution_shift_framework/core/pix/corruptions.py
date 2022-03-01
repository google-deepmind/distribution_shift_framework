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

"""Implementation of the ImageNet-C corruptions for sanity checks and eval.


All severity values are taken from ImageNet-C at
https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
"""

import chex
from distribution_shift_framework.core.pix import color_conversion
import dm_pix
import jax
import jax.numpy as jnp
import numpy as np


def scale_image(image: chex.Array, z_factor: chex.Numeric) -> chex.Array:
  """Resizes an image."""

  # And then resize
  b, h, w, c = image.shape
  resize_x = jax.image.scale_and_translate(
      image,
      shape=(b, int(h * z_factor), int(w * z_factor), c),
      method='bilinear',
      antialias=False,
      scale=jnp.ones((2,)) * z_factor,
      translation=jnp.zeros((2,)),
      spatial_dims=(1, 2))

  return resize_x


def zoom_blur(image: chex.Array, severity: int = 1, rng: chex.PRNGKey = None
              ) -> chex.Array:
  """The zoom blur corruption from ImageNet-C."""
  del rng

  c = [
      np.arange(1, 1.11, 0.01),
      np.arange(1, 1.16, 0.01),
      np.arange(1, 1.21, 0.02),
      np.arange(1, 1.26, 0.02),
      np.arange(1, 1.31, 0.03)
  ][severity - 1]

  _, h, w, _ = image.shape
  image_zoomed = jnp.zeros_like(image)
  for zoom_factor in c:
    t_image_zoomed = scale_image(image, zoom_factor)

    b = int(h * zoom_factor - h) // 2
    t_image_zoomed = t_image_zoomed[:, b:b + h, b:b + w, :]
    image_zoomed += t_image_zoomed

  image_zoomed = (image_zoomed + image) / (c.shape[0] + 1)
  return image_zoomed


def gaussian_blur(image: chex.Array,
                  severity: int = 1,
                  rng: chex.PRNGKey = None) -> chex.Array:
  """Gaussian blur corruption for ImageNet-C."""
  del rng
  c = [1, 2, 3, 4, 6][severity - 1]
  return dm_pix.gaussian_blur(image, sigma=c, kernel_size=image.shape[1])


def speckle_noise(image: chex.Array,
                  severity: int = 1,
                  rng: chex.PRNGKey = None) -> chex.Array:
  """Speckle noise corruption in ImageNet-C."""
  c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

  image = image + image * jax.random.normal(rng, shape=image.shape) * c
  return jnp.clip(image, a_min=0, a_max=1)


def impulse_noise(image: chex.Array,
                  severity: int = 1,
                  rng: chex.PRNGKey = None) -> chex.Array:
  """Impulse noise corruption in ImageNet-C."""
  c = [.03, .06, .09, 0.17, 0.27][severity - 1]
  x = jnp.clip(image, 0, 1)
  p = c
  q = 0.5
  out = x

  flipped = jax.random.choice(
      rng, 2, shape=x.shape, p=jax.numpy.array([1 - p, p]))
  salted = jax.random.choice(
      rng, 2, shape=x.shape, p=jax.numpy.array([1 - q, q]))
  peppered = 1 - salted

  mask = flipped * salted
  out = out * (1 - mask) + mask

  mask = flipped * peppered
  out = out * (1 - mask)
  return jnp.clip(out, a_min=0, a_max=1)


def shot_noise(image: chex.Array, severity: int = 1, rng: chex.PRNGKey = None
               ) -> chex.Array:
  """Shot noise in ImageNet-C corruptions."""
  c = [60, 25, 12, 5, 3][severity - 1]

  x = jnp.clip(image, 0, 1)
  x = jax.random.poisson(rng, lam=x * c, shape=x.shape) / c
  return jnp.clip(x, a_min=0, a_max=1)


def gaussian_noise(image: chex.Array,
                   severity: int = 1,
                   rng: chex.PRNGKey = None) -> chex.Array:
  """Gaussian noise in ImageNet-C corruptions."""
  c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

  x = image + jax.random.normal(rng, shape=image.shape) * c
  return jnp.clip(x, a_min=0, a_max=1)


def brightness(image: chex.Array, severity: int = 1, rng: chex.PRNGKey = None
               ) -> chex.Array:
  """The brightness corruption from ImageNet-C."""
  del rng
  c = [.1, .2, .3, .4, .5][severity - 1]

  x = jnp.clip(image, 0, 1)
  hsv = color_conversion.rgb_to_hsv(x)
  h, s, v = color_conversion.split_channels(hsv, -1)
  v = jnp.clip(v + c, 0, 1)
  rgb_adjusted = color_conversion.hsv_planes_to_rgb_planes(h, s, v)
  rgb = jnp.stack(rgb_adjusted, axis=-1)

  return rgb


def saturate(image: chex.Array, severity: int = 1, rng: chex.PRNGKey = None
             ) -> chex.Array:
  """The saturation corruption from ImageNet-C."""
  del rng
  c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

  x = jnp.clip(image, 0, 1)
  hsv = color_conversion.rgb_to_hsv(x)
  h, s, v = color_conversion.split_channels(hsv, -1)
  s = jnp.clip(s * c[0] + c[1], 0, 1)
  rgb_adjusted = color_conversion.hsv_planes_to_rgb_planes(h, s, v)
  rgb = jnp.stack(rgb_adjusted, axis=-1)

  return rgb


def contrast(image: chex.Array, severity: int = 1, rng: chex.PRNGKey = None
             ) -> chex.Array:
  """The contrast corruption from ImageNet-C."""
  del rng
  c = [0.4, .3, .2, .1, .05][severity - 1]

  return dm_pix.adjust_contrast(image, factor=c)
