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

"""Color conversion utilities.

These used to be in the dm_pix library but have been removed. I've added them
back here for the time being.
"""

from typing import Tuple

import chex
import jax.numpy as jnp


def split_channels(
    image: chex.Array,
    channel_axis: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  chex.assert_axis_dimension(image, axis=channel_axis, expected=3)
  split_axes = jnp.split(image, 3, axis=channel_axis)
  return tuple(map(lambda x: jnp.squeeze(x, axis=channel_axis), split_axes))


def rgb_to_hsv(
    image_rgb: chex.Array,
    *,
    channel_axis: int = -1,
) -> chex.Array:
  """Converts an image from RGB to HSV.

  Args:
    image_rgb: an RGB image, with float values in range [0, 1]. Behavior outside
      of these bounds is not guaranteed.
    channel_axis: the channel axis. image_rgb should have 3 layers along this
      axis.

  Returns:
    An HSV image, with float values in range [0, 1], stacked along channel_axis.
  """
  red, green, blue = split_channels(image_rgb, channel_axis)
  return jnp.stack(
      rgb_planes_to_hsv_planes(red, green, blue), axis=channel_axis)


def rgb_planes_to_hsv_planes(
    red: chex.Array,
    green: chex.Array,
    blue: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  """Converts red, green, blue color planes to hue, saturation, value planes.

  All planes should have the same shape, with float values in range [0, 1].
  Behavior outside of these bounds is not guaranteed.

  Reference implementation:
  https://github.com/tensorflow/tensorflow/blob/262f4ad303c78a99e0974c4b17892db2255738a0/tensorflow/compiler/tf2xla/kernels/image_ops.cc#L36-L68

  Args:
    red: the red color plane.
    green: the red color plane.
    blue: the red color plane.

  Returns:
    A tuple of (hue, saturation, value) planes, as float values in range [0, 1].
  """
  value = jnp.maximum(jnp.maximum(red, green), blue)
  minimum = jnp.minimum(jnp.minimum(red, green), blue)
  range_ = value - minimum

  saturation = jnp.where(value > 0, range_ / value, 0.)
  norm = 1. / (6. * range_)

  hue = jnp.where(value == green,
                  norm * (blue - red) + 2. / 6.,
                  norm * (red - green) + 4. / 6.)
  hue = jnp.where(value == red, norm * (green - blue), hue)
  hue = jnp.where(range_ > 0, hue, 0.) + (hue < 0.)

  return hue, saturation, value


def hsv_planes_to_rgb_planes(
    hue: chex.Array,
    saturation: chex.Array,
    value: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  """Converts hue, saturation, value planes to red, green, blue color planes.

  All planes should have the same shape, with float values in range [0, 1].
  Behavior outside of these bounds is not guaranteed.

  Reference implementation:
  https://github.com/tensorflow/tensorflow/blob/262f4ad303c78a99e0974c4b17892db2255738a0/tensorflow/compiler/tf2xla/kernels/image_ops.cc#L71-L94

  Args:
    hue: the hue plane (wrapping).
    saturation: the saturation plane.
    value: the value plane.

  Returns:
    A tuple of (red, green, blue) planes, as float values in range [0, 1].
  """
  dh = (hue % 1.0) * 6.  # Wrap when hue >= 360°.
  dr = jnp.clip(jnp.abs(dh - 3.) - 1., 0., 1.)
  dg = jnp.clip(2. - jnp.abs(dh - 2.), 0., 1.)
  db = jnp.clip(2. - jnp.abs(dh - 4.), 0., 1.)
  one_minus_s = 1. - saturation

  red = value * (one_minus_s + saturation * dr)
  green = value * (one_minus_s + saturation * dg)
  blue = value * (one_minus_s + saturation * db)

  return red, green, blue
