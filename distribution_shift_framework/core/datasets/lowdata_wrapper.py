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

"""Generates the low data versions for the disentanglement datasets."""
from typing import Callable, Optional, Sequence

from distribution_shift_framework.core.datasets import data_utils
import jax
import ml_collections
import tensorflow.compat.v2 as tf


def _create_filter_fn(filter_string: str) ->  Callable[..., bool]:
  """Creates a filter function based on the string.

  Given a string of
    "key11:val11:comp11^key12:val12:comp1b|...^keyNk:valNk:compNk"
  the string is parsed as the OR of several AND statements. The ORs are at the
  top level (denoted by |), and divide into a set of AND statements.
  The AND values are denoted by ^ and operate at the bottom level.
  Fof each "keyij:valij:compij" pairing, keyij is the key in the dataset,
  valij is the value the key is compared against and compij is the tensorflow
  comparison function: e.g. less, less_equal, equal, greater_equal, greater.

  Note that parentheses and infinite depth are *not* supported yet.

  Example 1: for dSprites: "label_scale:3:equal".
  This will select all samples from dSprites where the label_scale parameter is
  equal to 3.

  Example 2: for Shapes3D:
    "wall_hue_value:0.3:less_equal^floor_hue_value:0.3:less_equal".
  This will select all samples from Shapes3D where the wall hue and floor hue
  are less than or equal to 0.3.

  Example 3: for smallNORB:
    ('label_azimuth:7:less^label_category:0:equal|'
     'label_azimuth:7:greater_equal^label_category:0:not_equal').
  This will select all samples from smallNORB which either have azimuth of less
  than 7 and category 0 or azimuth of greater or equal 7 and a category other
  than 0.

  Args:
    filter_string: The filter string that is used to make the filter function.

  Returns:
    filter_fn: A function that takes a batch and returns True or False if it
      matches the filter string.
  """
  all_comparisons = filter_string.split('|')

  def filter_fn(x):
    or_filter = False

    # Iterate over all the OR comparisons.
    for or_comparison in all_comparisons:
      and_comparisons = or_comparison.split('^')

      and_filter = True
      # Iterate over all the AND comparisons.
      for and_comparison in and_comparisons:
        key, value, comp = and_comparison.split(':')
        if value in x.keys():
          value = x[value]
        else:
          value = tf.cast(float(value), x[key].dtype)
        bool_fn = getattr(tf, comp)
        # Accumulate the and comparisons.
        and_filter = tf.logical_and(and_filter, bool_fn(x[key], value))
      # Accumulate the or comparisons.
      or_filter = tf.logical_or(or_filter, and_filter)

    return or_filter

  return filter_fn


def load_data(batch_sizes: Sequence[int],
              dataset_loader: Callable[..., data_utils.Dataset],
              num_samples: str,
              filter_fns: str,
              dataset_kwargs: ml_collections.ConfigDict,
              shuffle_pre_sampling: bool = False,
              shuffle_pre_sample_seed: int = 0,
              local_cache: bool = True,
              is_training: bool = True,
              transpose: bool = True,
              drop_remainder: bool = True,
              prefilter: Optional[Callable[..., bool]] = None,
              preprocess_fn: Optional[data_utils.PreprocessFnGen] = None,
              shuffle_buffer: Optional[int] = 100_000,
              weights: Optional[Sequence[float]] = None) -> data_utils.Dataset:
  """A low data wrapper around a tfds dataset.

  This wrapper creates a set of datasets according to the parameters. For each
  filtering function and number of samples, the dataset defined by the
  dataset_loader and **dataset_kwargs is filtered and the first N samples are
  taken. All datasets are concatenated together and a sample is drawn with
  equal probability from each dataset.

  Args:
    batch_sizes: Specifies how to batch examples. I.e., if batch_sizes = [8, 4]
      then output images will have shapes (8, 4, height, width, 3).
    dataset_loader: The tfds dataset loader.
    num_samples: An string of the number of samples each returned dataset will
      contain.  I.e., if num_samples = '1,2,3' then the first filtering
      operation will create a dataset with 1 sample, the second a dataset of 2
      samples, and so on.
    filter_fns: An iterable of the filtering functions for each part of the
      dataset.
    dataset_kwargs: A dict of the kwargs to pass to dataset_loader.
    shuffle_pre_sampling: Whether to shuffle presampling and thereby get a
      different set of samples.
    shuffle_pre_sample_seed: What seed to use for presampling.
    local_cache: Whether to cache the concatenated dataset. Good to do if the
      dataset fits in memory.
    is_training: Whether this is train or test.
    transpose: Whether to permute image dimensions NHWC -> HWCN to speed up
      performance on TPUs.
    drop_remainder: Whether to drop the last batch(es) if they would not match
      the shapes specified by `batch_sizes`.
    prefilter: Filter to apply to the dataset.
    preprocess_fn: Function mapped onto each example for pre-processing.
    shuffle_buffer: How big the buffer for shuffling the images is.
    weights: The probabilities to select samples from each dataset.

  Returns:
    A tf.Dataset instance.
  """
  ds = dataset_loader(**dataset_kwargs)

  if preprocess_fn:
    ds = ds.map(
        preprocess_fn('train' if is_training else 'test'),
        num_parallel_calls=tf.data.AUTOTUNE)

  if prefilter:
    ds.filter(prefilter)

  filter_fns = filter_fns.split(',')
  num_samples = [int(n) for n in num_samples.split(',')]

  assert len(filter_fns) == len(num_samples)

  all_ds = []
  for filter_fn, n_sample in zip(filter_fns, num_samples):
    if filter_fn != 'True':
      filter_fn = _create_filter_fn(filter_fn)
      filtered_ds = ds.filter(filter_fn)
    else:
      filtered_ds = ds

    if shuffle_pre_sampling:
      filtered_ds = filtered_ds.shuffle(
          buffer_size=shuffle_buffer, seed=shuffle_pre_sample_seed)

    if n_sample:
      filtered_ds = filtered_ds.take(n_sample)
    if local_cache or n_sample:
      filtered_ds = filtered_ds.cache()

    if is_training:
      filtered_ds = filtered_ds.repeat()

      shuffle_buffer = (
          min(n_sample, shuffle_buffer) if n_sample > 0 else shuffle_buffer)
      filtered_ds = filtered_ds.shuffle(
          buffer_size=shuffle_buffer, seed=jax.process_index())
    all_ds.append(filtered_ds)

  ds = tf.data.Dataset.sample_from_datasets(
      all_ds, weights=weights, seed=None)

  for i, batch_size in enumerate(reversed(batch_sizes)):
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if i == 0 and transpose:
      ds = ds.map(data_utils.transpose_fn)  # NHWC -> HWCN.

  return ds
