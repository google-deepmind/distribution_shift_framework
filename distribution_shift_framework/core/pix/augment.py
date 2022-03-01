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

"""This module provides image augmentation functions.

All functions expect float-encoded images, with values between 0 and 1, but
do not clip their outputs.
"""

import chex
from distribution_shift_framework.core.pix import color_conversion
import jax
import jax.numpy as jnp


def _auto_contrast(image: chex.Array, cutoff: int = 0) -> chex.Array:
  """The auto contrast transform: remove top/bottom % and rescale histogram.

  Args:
    image: an RGB image given as a float tensor in [0, 1].
    cutoff: what % of higher/lower pixels to remove

  Returns:
    The new image with auto contrast applied.
  """
  im_rgbs = []
  indices = jnp.arange(0, 256, 1)
  for rgb in range(0, image.shape[2]):
    im_rgb = image[:, :, rgb:rgb + 1]
    hist = jnp.histogram(im_rgb, bins=256, range=(0, 1))[0]

    hist_cumsum = hist.cumsum()
    # Determine % samples
    cut_lower = hist_cumsum[-1] * cutoff // 100
    cut_higher = hist_cumsum[-1] * (100 - cutoff) // 100

    # The lower offset
    offset_lo = (hist_cumsum < cut_lower) * indices
    offset_lo = offset_lo.max() / 256.

    # The higher offset
    offset_hi = (hist_cumsum <= cut_higher) * indices
    offset_hi = offset_hi.max() / 256.

    # Remove cutoff% samples from low/hi end
    im_rgb = (im_rgb - offset_lo).clip(0, 1) + offset_lo
    im_rgb = (im_rgb + 1 - offset_hi).clip(0, 1) - (1 - offset_hi)

    # And renormalize
    offset = (offset_hi - offset_lo) < 1 / 256.
    im_rgb = (im_rgb - offset_lo) / (offset_hi - offset_lo + offset)

    # And return
    im_rgbs.append(im_rgb)

  return jnp.concatenate(im_rgbs, axis=2)


def auto_contrast(image: chex.Array, cutoff: chex.Array) -> chex.Array:
  if len(image.shape) < 4:
    return _auto_contrast(image, cutoff)

  else:
    return jax.vmap(_auto_contrast)(image, cutoff.astype(jnp.int32))


def _equalize(image: chex.Array) -> chex.Array:
  """The equalize transform: make histogram cover full scale.

  Args:
    image: an RGB image given as a float tensor in [0, 1].

  Returns:
    The equalized image.
  """
  im_rgbs = []

  im = (image * 255).astype(jnp.int32).clip(0, 255)
  for rgb in range(0, im.shape[2]):
    im_rgb = im[:, :, rgb:rgb + 1]

    hist = jnp.histogram(im_rgb, bins=256, range=(0, 256))[0]

    last_nonzero_value = hist.sum() - hist.cumsum()
    last_nonzero_value = last_nonzero_value + last_nonzero_value.max() * (
        last_nonzero_value == 0)
    step = (hist.sum() - last_nonzero_value.min()) // 255
    n = step // 2

    im_rgb_new = jnp.zeros((im_rgb.shape), dtype=im_rgb.dtype)

    def for_loop(i, values):
      (im, n, hist, step, im_rgb) = values
      im = im + (n // step) * (im_rgb == i)

      return (im, n + hist[i], hist, step, im_rgb)

    result, _, _, _, _ = jax.lax.fori_loop(0, 256, for_loop,
                                           (im_rgb_new, n, hist, step, im_rgb))

    im_rgbs.append(result.astype(jnp.float32) / 255.)
  return jnp.concatenate(im_rgbs, 2)


def equalize(image: chex.Array, unused_cutoff: chex.Array) -> chex.Array:
  if len(image.shape) < 4:
    return _equalize(image)
  else:
    return jax.vmap(_equalize)(image)


def _posterize(image: chex.Array, bits: chex.Array) -> chex.Array:
  """The posterize transform: remove least significant bits.

  Args:
    image: an RGB image given as a float tensor in [0, 1].
    bits: how many bits to ignore.

  Returns:
    The posterized image.
  """
  mask = ~(2**(8 - bits) - 1)
  image = (image * 255).astype(jnp.int32).clip(0, 255)

  image = jnp.bitwise_and(image, mask)
  return image.astype(jnp.float32) / 255.


def posterize(image: chex.Array, bits: chex.Array) -> chex.Array:
  if len(image.shape) < 4:
    return _posterize(image, bits)
  else:
    return jax.vmap(_posterize)(image, bits.astype(jnp.uint8))


def _solarize(image: chex.Array, threshold: chex.Array) -> chex.Array:
  """The solarization transformation: pixels > threshold are inverted.

  Args:
    image: an RGB image given as a float tensor in [0, 1].
    threshold: the threshold in [0, 1] above which to invert the image.

  Returns:
    The solarized image.
  """
  image = (1 - image) * (image >= threshold) + image * (image < threshold)
  return image


def solarize(image: chex.Array, threshold: chex.Array) -> chex.Array:
  if len(image.shape) < 4:
    return _solarize(image, threshold)
  else:
    return jax.vmap(_solarize)(image, threshold)


def adjust_color(image: chex.Array,
                 factor: chex.Numeric,
                 channel: int = 0,
                 channel_axis: int = -1) -> chex.Array:
  """Shifts the color of an RGB by a given multiplicative amount.

  Args:
    image: an RGB image, given as a float tensor in [0, 1].
    factor: the (additive) amount to shift the RGB by.
    channel: the RGB channel to manipulate
    channel_axis: the index of the channel axis.

  Returns:
    The color adjusted image.
  """
  red, green, blue = color_conversion.split_channels(image, channel_axis)

  if channel == 0:
    red = jnp.clip(red + factor, 0., 1.)
  elif channel == 1:
    green = jnp.clip(green + factor, 0., 1.)
  else:
    blue = jnp.clip(blue + factor, 0., 1.)

  return jnp.stack((red, green, blue), axis=channel_axis)
