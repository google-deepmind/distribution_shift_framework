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

"""Loader and preprocessing functions for the datasets."""

from typing import Optional, Sequence

import chex
from distribution_shift_framework.core.datasets import data_utils
import jax
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def shapes3d_normalize(image: chex.Array) -> chex.Array:
  return (image - .5) * 2


def shapes3d_preprocess(
    mode: str = 'train'
) -> data_utils.TFPreprocessFn:
  del mode
  def _preprocess_fn(example):
    example['image'] = tf.image.convert_image_dtype(
        example['image'], dtype=tf.float32)
    example['label'] = example['label_shape']
    return example
  return _preprocess_fn


def unbatched_load_shapes3d(subset: str = 'train',
                            valid_size: int = 10000,
                            test_size: int = 10000) -> data_utils.Dataset:
  """Loads the 3D Shapes dataset without batching."""
  if subset == 'train':
    ds = tfds.load(name='shapes3d', split=tfds.Split.TRAIN
                   ).skip(valid_size + test_size)
  elif subset == 'valid':
    ds = tfds.load(name='shapes3d', split=tfds.Split.TRAIN
                   ).skip(test_size).take(valid_size)
  elif subset == 'train_and_valid':
    ds = tfds.load(name='shapes3d', split=tfds.Split.TRAIN).skip(test_size)
  elif subset == 'test':
    ds = tfds.load(name='shapes3d', split=tfds.Split.TRAIN).take(test_size)
  else:
    raise ValueError('Unknown subset: "{}"'.format(subset))
  return ds


def load_shapes3d(batch_sizes: Sequence[int],
                  subset: str = 'train',
                  is_training: bool = True,
                  num_samples: Optional[int] = None,
                  preprocess_fn: Optional[data_utils.PreprocessFnGen] = None,
                  transpose: bool = False,
                  valid_size: int = 10000,
                  test_size: int = 10000,
                  drop_remainder: bool = True,
                  local_cache: bool = True) -> data_utils.Dataset:
  """Loads the 3D Shapes dataset.

  The 3D shapes dataset is available at https://github.com/deepmind/3d-shapes.
  It consists of 4 different shapes which vary along 5 different axes:
  - Floor hue: 10 colors with varying red, orange, yellow, green, blue
  - Wall hue: 10 colors with varying red, orange, yellow, green, blue
  - Object hue: 10 colors with varying red, orange, yellow, green, blue
  - Scale: How large the object is.
  - Shape: 4 values -- (cube, sphere, cylinder, and oblong).
  - Orientation: Rotates the object around the vertical axis.

  Args:
    batch_sizes: Specifies how to batch examples. I.e., if batch_sizes = [8, 4]
      then output images will have shapes (8, 4, height, width, 3).
    subset: Specifies which subset (train, valid or train_and_valid) to use.
    is_training: Whether to infinitely repeat and shuffle examples (`True`) or
      not (`False`).
    num_samples: The number of samples to crop each individual dataset variant
      from the start, or `None` to use the full dataset.
    preprocess_fn: Function mapped onto each example for pre-processing.
    transpose: Whether to permute image dimensions NHWC -> HWCN to speed up
      performance on TPUs.
    valid_size: Size of the validation set to take from the training set.
    test_size: Size of the validation set to take from the training set.
    drop_remainder: Whether to drop the last batch(es) if they would not match
      the shapes specified by `batch_sizes`.
    local_cache: Whether to locally cache the dataset.

  Returns:
    ds: Fully configured dataset ready for training/evaluation.
  """
  if preprocess_fn is None:
    preprocess_fn = shapes3d_preprocess
  ds = unbatched_load_shapes3d(subset=subset, valid_size=valid_size,
                               test_size=test_size)
  total_batch_size = np.prod(batch_sizes)
  if subset == 'valid' and valid_size < total_batch_size:
    ds = ds.repeat().take(total_batch_size)
  ds = batch_and_shuffle(ds, batch_sizes,
                         is_training=is_training,
                         transpose=transpose,
                         num_samples=num_samples,
                         preprocess_fn=preprocess_fn,
                         drop_remainder=drop_remainder,
                         local_cache=local_cache)
  return ds


def small_norb_normalize(image: chex.Array) -> chex.Array:
  return (image - .5) * 2


def small_norb_preprocess(
    mode: str = 'train'
) -> data_utils.TFPreprocessFn:
  del mode
  def _preprocess_fn(example):
    example['image'] = tf.image.convert_image_dtype(
        example['image'], dtype=tf.float32)
    example['label'] = example['label_category']
    return example
  return _preprocess_fn


def unbatched_load_small_norb(subset: str = 'train',
                              valid_size: int = 10000) -> data_utils.Dataset:
  """Load the small norb dataset."""
  if subset == 'train':
    ds = tfds.load(name='smallnorb', split=tfds.Split.TRAIN).skip(valid_size)
  elif subset == 'valid':
    ds = tfds.load(name='smallnorb', split=tfds.Split.TRAIN).take(valid_size)
  elif subset == 'train_and_valid':
    ds = tfds.load(name='smallnorb', split=tfds.Split.TRAIN)
  elif subset == 'test':
    ds = tfds.load(name='smallnorb', split=tfds.Split.TEST)
  else:
    raise ValueError('Unknown subset: "{}"'.format(subset))
  return ds


def load_small_norb(batch_sizes: Sequence[int],
                    subset: str = 'train',
                    is_training: bool = True,
                    num_samples: Optional[int] = None,
                    preprocess_fn: Optional[data_utils.PreprocessFnGen] = None,
                    transpose: bool = False,
                    valid_size: int = 1000,
                    drop_remainder: bool = True,
                    local_cache: bool = True) -> data_utils.Dataset:
  """Loads the small norb dataset.

  The norb dataset is available at:
    https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/.

  It consists of 5 categories (Animals, People, Airplanes, Trucks, and Cars).
  These categories have 5 instances (different animals, airplanes, or types of
  cars).

  They vary by (which are consistent across categories and instances):
  1. Elevation
  2. Azimuth
  3. Lighting

  Args:
    batch_sizes: Specifies how to batch examples. I.e., if batch_sizes = [8, 4]
      then output images will have shapes (8, 4, height, width, 3).
    subset: Specifies the subset (train, valid, test or train_and_valid) to use.
    is_training: Whether to infinitely repeat and shuffle examples (`True`) or
      not (`False`).
    num_samples: The number of samples to crop each individual dataset variant
      from the start, or `None` to use the full dataset.
    preprocess_fn: Function mapped onto each example for pre-processing.
    transpose: Whether to permute image dimensions NHWC -> HWCN to speed up
      performance on TPUs.
    valid_size: The size of the validation set.
    drop_remainder: Whether to drop the last batch(es) if they would not match
      the shapes specified by `batch_sizes`.
    local_cache: Whether to locally cache the dataset.

  Returns:
    ds: Fully configured dataset ready for training/evaluation.
  """
  if preprocess_fn is None:
    preprocess_fn = small_norb_preprocess

  ds = unbatched_load_small_norb(subset=subset, valid_size=valid_size)
  total_batch_size = np.prod(batch_sizes)
  if subset == 'valid' and valid_size < total_batch_size:
    ds = ds.repeat().take(total_batch_size)
  ds = batch_and_shuffle(ds, batch_sizes,
                         is_training=is_training,
                         transpose=transpose,
                         num_samples=num_samples,
                         preprocess_fn=preprocess_fn,
                         drop_remainder=drop_remainder,
                         local_cache=local_cache)
  return ds


def dsprites_normalize(image: chex.Array) -> chex.Array:
  return (image - .5) * 2


def dsprites_preprocess(
    mode: str = 'train'
) -> data_utils.TFPreprocessFn:
  del mode
  def _preprocess_fn(example):
    example['image'] = tf.image.convert_image_dtype(
        example['image'], dtype=tf.float32) * 255.
    example['label'] = example['label_shape']
    return example
  return _preprocess_fn


def unbatched_load_dsprites(subset: str = 'train',
                            valid_size: int = 10000,
                            test_size: int = 10000) -> data_utils.Dataset:
  """Loads the dsprites dataset without batching and prefetching."""
  if subset == 'train':
    ds = tfds.load(name='dsprites',
                   split=tfds.Split.TRAIN).skip(valid_size + test_size)
  elif subset == 'valid':
    ds = tfds.load(name='dsprites',
                   split=tfds.Split.TRAIN).skip(test_size).take(valid_size)
  elif subset == 'train_and_valid':
    ds = tfds.load(name='dsprites', split=tfds.Split.TRAIN).skip(test_size)
  elif subset == 'test':
    ds = tfds.load(name='dsprites', split=tfds.Split.TRAIN).take(test_size)
  else:
    raise ValueError('Unknown subset: "{}"'.format(subset))
  return ds


def load_dsprites(batch_sizes: Sequence[int],
                  subset: str = 'train',
                  is_training: bool = True,
                  num_samples: Optional[int] = None,
                  preprocess_fn: Optional[data_utils.PreprocessFnGen] = None,
                  transpose: bool = False,
                  valid_size: int = 10000,
                  test_size: int = 10000,
                  drop_remainder: bool = True,
                  local_cache: bool = True) -> data_utils.Dataset:
  """Loads the dsprites dataset.

  The dsprites dataset is available at:
    https://github.com/deepmind/dsprites-dataset.

  It consists of 3 shapes (heart, ellipse and square).

  They vary by (which are consistent across categories and instances):
  1. Scale (6 values)
  2. Orientation: 40 values (rotates around the center of the object)
  3. Position (X): 32 values
  4. Position (Y): 32 values

  Args:
    batch_sizes: Specifies how to batch examples. I.e., if batch_sizes = [8, 4]
      then output images will have shapes (8, 4, height, width, 3).
    subset: Specifies the subset (train, valid, test or train_and_valid) to use.
    is_training: Whether to infinitely repeat and shuffle examples (`True`) or
      not (`False`).
    num_samples: The number of samples to crop each individual dataset variant
      from the start, or `None` to use the full dataset.
    preprocess_fn: Function mapped onto each example for pre-processing.
    transpose: Whether to permute image dimensions NHWC -> HWCN to speed up
      performance on TPUs.
    valid_size: The size of the validation set.
    test_size: The size of the test set.
    drop_remainder: Whether to drop the last batch(es) if they would not match
      the shapes specified by `batch_sizes`.
    local_cache: Whether to locally cache the dataset.

  Returns:
    ds: Fully configured dataset ready for training/evaluation.
  """
  if preprocess_fn is None:
    preprocess_fn = dsprites_preprocess

  ds = unbatched_load_dsprites(subset=subset, valid_size=valid_size,
                               test_size=test_size)
  total_batch_size = np.prod(batch_sizes)
  if subset == 'valid' and valid_size < total_batch_size:
    ds = ds.repeat().take(total_batch_size)
  ds = batch_and_shuffle(ds, batch_sizes,
                         is_training=is_training,
                         transpose=transpose,
                         num_samples=num_samples,
                         preprocess_fn=preprocess_fn,
                         drop_remainder=drop_remainder,
                         local_cache=local_cache)
  return ds


def batch_and_shuffle(
    ds: data_utils.Dataset,
    batch_sizes: Sequence[int],
    preprocess_fn: Optional[data_utils.PreprocessFnGen] = None,
    is_training: bool = True,
    num_samples: Optional[int] = None,
    transpose: bool = False,
    drop_remainder: bool = True,
    local_cache: bool = False) -> data_utils.Dataset:
  """Performs post-processing on datasets (i.e., batching, transposing).

  Args:
    ds: The dataset.
    batch_sizes: Specifies how to batch examples. I.e., if batch_sizes = [8, 4]
      then output images will have shapes (8, 4, height, width, 3).
    preprocess_fn: Function mapped onto each example for pre-processing.
    is_training: Whether to infinitely repeat and shuffle examples (`True`) or
      not (`False`).
    num_samples: The number of samples to crop each individual dataset variant
      from the start, or `None` to use the full dataset.
    transpose: Whether to permute image dimensions NHWC -> HWCN to speed up
      performance on TPUs.
    drop_remainder:  Whether to drop the last batch(es) if they would not match
      the shapes specified by `batch_sizes`.
    local_cache: Whether to locally cache the dataset.
  Returns:
    ds: Dataset with all the post-processing applied.
  """
  if num_samples:
    ds = ds.take(num_samples)
  if local_cache:
    ds = ds.cache()
  if is_training:
    ds = ds.repeat()
    total_batch_size = np.prod(batch_sizes)
    shuffle_buffer = 10 * total_batch_size
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=jax.process_index())
  if preprocess_fn is not None:
    ds = ds.map(preprocess_fn('train' if is_training else 'test'),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  for i, batch_size in enumerate(reversed(batch_sizes)):
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    if i == 0 and transpose:
      ds = ds.map(data_utils.transpose_fn)  # NHWC -> HWCN.
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds
