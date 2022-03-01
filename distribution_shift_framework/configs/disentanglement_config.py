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

"""Configs for disentanglement datasets."""
import itertools
from typing import Any, Callable, Optional, Sequence

from distribution_shift_framework.core import hyper
from distribution_shift_framework.core.datasets import data_loaders
from distribution_shift_framework.core.datasets import data_utils
from distribution_shift_framework.core.datasets import lowdata_wrapper
import ml_collections
import tensorflow.compat.v2 as tf


_VALID_COLORS = ((
    1,
    0,
    0,
), (0, 1, 0), (0, 0, 1))
_EXP = 'config.experiment_kwargs.config'
_TRAIN_SPLIT = 'train'
_TEST_SPLIT = 'valid'

_ExampleFn = Callable[[tf.train.Example], tf.train.Example]


def _color_preprocess(mode: str,
                      preprocess: Optional[Callable[[str], _ExampleFn]] = None,
                      label: str = 'label') -> _ExampleFn:
  """Preprocessing function to add colour to white pixels in a binary image."""

  def _color_fn(example: tf.train.Example) -> tf.train.Example:
    if preprocess is not None:
      example = preprocess(mode)(example)

    example['image'] = tf.repeat(example['image'], 3, axis=2)
    example['image'] = tf.cast(example['image'], tf.float32)

    # Choose a random color.
    color_id = tf.random.uniform(
        shape=(), minval=0, maxval=len(_VALID_COLORS), dtype=tf.int64)
    example['label_color'] = color_id

    colors = tf.constant(_VALID_COLORS, dtype=tf.float32)[color_id]
    example['image'] = example['image'] * colors
    example['label'] = example[label]

    example['fairness_features'] = {
        k: v for k, v in example.items() if k.startswith('label_')
    }
    return example

  return _color_fn


def _get_base_config(dataset_name: str, label: str, property_label: str
                     ) -> ml_collections.ConfigDict:
  """Get base config."""
  data = ml_collections.ConfigDict()
  data.name = dataset_name
  data.test = dataset_name

  dataset_constants = data_utils.get_dataset_constants(dataset_name, label)
  data.label_property = property_label
  data.label = label

  data.n_classes = dataset_constants['num_classes']
  data.num_channels = dataset_constants['num_channels']
  data.image_size = dataset_constants['image_size']
  data.variance = dataset_constants['variance']

  if dataset_name != data_utils.DatasetNames.DSPRITES.value or (
      property_label != 'label_color'):
    data.prop_values = dataset_constants['properties'][property_label]
    data.n_properties = len(data.prop_values)

  if dataset_name == data_utils.DatasetNames.DSPRITES.value and (
      label == 'label_color' or property_label == 'label_color'):

    data.num_channels = 3

    if label == 'label_color':
      data.n_classes = 3
    if property_label == 'label_color':
      data.prop_values = (0, 1, 2)
      data.n_properties = 3

  return data


def _get_filter_fns(values: Sequence[Any],
                    perc_property: float,
                    property_name: str) -> str:
  cutoff = max(int((len(values) - 1) * perc_property), 0)
  cutoff = values[cutoff]
  filter_fns = (f'{property_name}:{cutoff}:less_equal,'
                f'{property_name}:{cutoff}:greater')
  return filter_fns


def get_data_config(dataset_name: str, label: str, property_label: str
                    ) -> ml_collections.ConfigDict:
  """Get config for a given setup."""
  data = _get_base_config(dataset_name, label, property_label)

  dataset_loader = getattr(data_loaders, f'unbatched_load_{dataset_name}', '')
  preprocess_fn = getattr(data_loaders, f'{dataset_name}_preprocess', '')
  full_dataset_loader = getattr(data_loaders, f'load_{dataset_name}', '')

  data.train_kwargs = ml_collections.ConfigDict()
  data.train_kwargs.loader = lowdata_wrapper.load_data

  data.train_kwargs.load_kwargs = dict()
  data.train_kwargs.load_kwargs.dataset_loader = dataset_loader
  data.train_kwargs.load_kwargs.weights = [1.]

  data.train_kwargs.load_kwargs.dataset_kwargs = dict(subset=_TRAIN_SPLIT)
  data.train_kwargs.load_kwargs.preprocess_fn = preprocess_fn

  # Set up filters and number of samples.
  data.train_kwargs.load_kwargs.num_samples = '0'
  # A string to define how the dataset is filtered (not a boolean value).
  data.train_kwargs.load_kwargs.filter_fns = 'True'

  data.test_kwargs = ml_collections.ConfigDict()
  data.test_kwargs.loader = full_dataset_loader
  data.test_kwargs.load_kwargs = dict(subset=_TEST_SPLIT)

  if dataset_name == data_utils.DatasetNames.DSPRITES.value and (
      label == 'label_color' or property_label == 'label_color'):
    # Make the images different colours, as opposed to block and white.
    preprocess = data.train_kwargs.load_kwargs.preprocess_fn
    data.train_kwargs.load_kwargs.preprocess_fn = (
        lambda m: _color_preprocess(m, preprocess, label))
    data.test_kwargs.load_kwargs.preprocess_fn = (
        lambda m: _color_preprocess(m, None, label))

  return data


def get_alldata_config(dataset_name: str, label: str, property_label: str
                       ) -> ml_collections.ConfigDict:
  """Config when using the full dataset."""
  loader = getattr(data_loaders, f'load_{dataset_name}', '')
  data = _get_base_config(dataset_name, label, property_label)

  data.train_kwargs = ml_collections.ConfigDict()
  data.train_kwargs.loader = loader
  data.train_kwargs.load_kwargs = dict(subset=_TRAIN_SPLIT)

  data.test_kwargs = ml_collections.ConfigDict()
  data.test_kwargs.loader = loader
  data.test_kwargs.load_kwargs = dict(subset=_TEST_SPLIT)
  return data


def get_renderers(datatype: str,
                  dataset_name: str,
                  label: str,
                  property_label: str) -> ml_collections.ConfigDict:
  if len(datatype.split('.')) > 1:
    renderer, _ = datatype.split('.')
  else:
    renderer = datatype

  return globals()[f'get_{renderer}_renderers'](
      dataset_name, label=label, property_label=property_label)


def get_renderer_sweep(datatype: str) -> hyper.Sweep:
  if len(datatype.split('.')) > 1:
    _, sweep = datatype.split('.')
  else:
    sweep = datatype
  return globals()[f'get_{sweep}_sweep']()


def get_resample_sweep() -> hyper.Sweep:
  """Sweep over the resampling operation of the different datasets."""
  ratios = [1e-3]
  n_samples = [1_000_000]
  ratio_samples = list(itertools.product(ratios, n_samples))
  ratio_samples_sweep = hyper.sweep(
      f'{_EXP}.data.train_kwargs.load_kwargs.num_samples',
      [f'{n_s},{int(max(1, n_s * r))}' for r, n_s in ratio_samples])
  resample_weights = hyper.sweep(
      f'{_EXP}.data.train_kwargs.load_kwargs.weights',
      [[1 - i, i] for i in [1e-4, 1e-3, 1e-2, 1e-1, 0.5]])
  return hyper.product([ratio_samples_sweep, resample_weights])


def get_fixeddata_sweep() -> hyper.Sweep:
  """Sweep over the amount of data and noise present."""
  ratios = [1e-3]
  n_samples = [1000, 10_000, 100_000, 1_000_000]
  ratio_samples = list(itertools.product(ratios, n_samples))
  ratio_samples_sweep = hyper.sweep(
      f'{_EXP}.data.train_kwargs.load_kwargs.num_samples',
      [f'{n_s},{int(max(1, n_s * r))}' for r, n_s in ratio_samples])
  return ratio_samples_sweep


def get_noise_sweep() -> hyper.Sweep:
  return hyper.sweep(f'{_EXP}.training.label_noise',
                     [i / float(10.) for i in list(range(7, 11))])


def get_lowdata_sweep() -> hyper.Sweep:
  return hyper.sweep(
      f'{_EXP}.data.train_kwargs.load_kwargs.num_samples',
      [f'0,{n_s}' for n_s in [1, 5, 10, 50, 100, 500, 1000, 5000, 10_000]])


def get_ood_sweep() -> hyper.Sweep:
  return hyper.sweep(f'{_EXP}.data.train_kwargs.load_kwargs.weights',
                     [[1., 0.]])


def get_base_renderers(dataset_name: str,
                       label: str = 'color',
                       property_label: str = 'shape'
                       ) -> ml_collections.ConfigDict:
  """Get base config for the given dataset, label and property value."""
  data = get_data_config(dataset_name, label, property_label)
  data.train_kwargs.load_kwargs.filter_fns = 'True'
  data.train_kwargs.load_kwargs.num_samples = '0'
  data.train_kwargs.load_kwargs.weights = [1.]
  return data


def get_ood_renderers(dataset_name: str,
                      label: str = 'color',
                      property_label: str = 'shape'
                      ) -> ml_collections.ConfigDict:
  """Get OOD config for the given dataset, label and property value."""
  data = get_data_config(dataset_name, label, property_label)

  perc_props_in_train = 0.7 if dataset_name in ('dsprites') else 0.2
  data.train_kwargs.load_kwargs.filter_fns = _get_filter_fns(
      data.prop_values, perc_props_in_train, property_label)
  data.train_kwargs.load_kwargs.weights = [1., 0.]
  data.train_kwargs.load_kwargs.num_samples = '0,1000'
  return data


def get_correlated_renderers(dataset_name: str,
                             label: str = 'color',
                             property_label: str = 'shape'
                             ) -> ml_collections.ConfigDict:
  """Get correlated config for the given dataset, label and property value."""
  data = get_data_config(dataset_name, label, property_label)
  data.train_kwargs.load_kwargs.filter_fns = (
      f'{label}:{property_label}:equal,True')
  data.train_kwargs.load_kwargs.weights = [0.5, 0.5]
  num_samples = '0,500' if dataset_name == 'dsprites' else '0,50'
  data.train_kwargs.load_kwargs.num_samples = num_samples
  data.train_kwargs.load_kwargs.shuffle_pre_sampling = True
  data.train_kwargs.load_kwargs.shuffle_pre_sample_seed = 0
  return data


def get_lowdata_renderers(dataset_name: str,
                          label: str = 'color',
                          property_label: str = 'shape'
                          ) -> ml_collections.ConfigDict:
  """Get lowdata config for the given dataset, label and property value."""
  data = get_ood_renderers(dataset_name, label, property_label)
  data.train_kwargs.load_kwargs.weights = [0.5, 0.5]
  data.train_kwargs.load_kwargs.num_samples = '0,10'
  data.train_kwargs.load_kwargs.shuffle_pre_sampling = True
  data.train_kwargs.load_kwargs.shuffle_pre_sample_seed = 0

  return data
