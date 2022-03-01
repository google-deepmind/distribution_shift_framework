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

"""Data utility functions."""
import enum
from typing import Any, Callable, Dict, Generator, Iterable, Mapping, Optional, Sequence, Tuple, Union

import chex
import jax
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# Type Aliases
Batch = Dict[str, np.ndarray]
ScalarDict = Dict[str, chex.Array]
# Objects that can be treated like tensors in TF2.
TFTensorLike = Union[np.ndarray, tf.Tensor, tf.Variable]
# pytype: disable=not-supported-yet
TFTensorNest = Union[TFTensorLike, Iterable['TFTensorNest'],
                     Mapping[str, 'TFTensorNest']]
# pytype: enable=not-supported-yet
PreprocessFnGen = Callable[[str], Callable[[chex.ArrayTree], chex.ArrayTree]]
TFPreprocessFn = Callable[[TFTensorNest], TFTensorNest]

Dataset = tf.data.Dataset

# Disentanglement datasets.

SHAPES3D_PROPERTIES = {
    'label_scale': tuple(range(8)),
    'label_orientation': tuple(range(15)),
    'label_floor_hue': tuple(range(10)),
    'label_object_hue': tuple(range(10)),
    'label_wall_hue': tuple(range(10)),
    'label_shape': tuple(range(4)),
    'label_color':
        tuple(range(3))  # Only added through preprocessing.
}

SMALL_NORB_PROPERTIES = {
    'label_azimuth': tuple(range(18)),
    'label_elevation': tuple(range(9)),
    'label_lighting': tuple(range(6)),
    'label_category': tuple(range(5)),
}

DSPRITES_PROPERTIES = {
    'label_scale': tuple(range(6)),
    'label_orientation': tuple(range(40)),
    'label_x_position': tuple(range(32)),
    'label_y_position': tuple(range(32)),
    'label_shape': tuple(range(3)),
}


class DatasetNames(enum.Enum):
  """Names of the datasets."""
  SHAPES3D = 'shapes3d'
  SMALL_NORB = 'small_norb'
  DSPRITES = 'dsprites'


class NumChannels(enum.Enum):
  """Number of channels of the images."""
  SHAPES3D = 3
  SMALL_NORB = 1
  DSPRITES = 1


class Variance(enum.Enum):
  """Variance of the pixels in the images."""
  SHAPES3D = 0.155252
  SMALL_NORB = 0.031452
  DSPRITES = 0.04068864749147259


class ImageSize(enum.Enum):
  """Size of the images."""
  SHAPES3D = 64
  SMALL_NORB = 96
  DSPRITES = 64


def is_disentanglement_dataset(dataset_name: str) -> bool:
  return dataset_name in (DatasetNames.SHAPES3D.value,
                          DatasetNames.SMALL_NORB.value,
                          DatasetNames.DSPRITES.value)


def get_dataset_constants(dataset_name: str,
                          label: str = 'label',
                          variant: Optional[str] = None) -> Mapping[str, Any]:
  """Returns a dictionary with several constants for the dataset."""
  if variant:
    properties_name = f'{dataset_name.upper()}_{variant.upper()}_PROPERTIES'
  else:
    properties_name = f'{dataset_name.upper()}_PROPERTIES'
  properties = globals()[properties_name]
  num_channels = NumChannels[dataset_name.upper()].value

  if dataset_name == DatasetNames.DSPRITES.value and label == 'label_color':
    num_classes = 3
  else:
    num_classes = len(properties[label])

  return {
      'properties': properties,
      'num_channels': num_channels,
      'num_classes': num_classes,
      'variance': Variance[dataset_name.upper()].value,
      'image_size': ImageSize[dataset_name.upper()].value
  }


def transpose_fn(batch: Batch) -> Batch:
  # Transpose for performance on TPU.
  batch = dict(**batch)
  batch['image'] = tf.transpose(batch['image'], (1, 2, 3, 0))
  return batch


def load_dataset(is_training: bool,
                 batch_dims: Sequence[int],
                 transpose: bool,
                 data_kwargs: Optional[ml_collections.ConfigDict] = None
                 ) -> Generator[Batch, None, None]:
  """Wrapper to load a dataset."""

  data_loader = data_kwargs['loader']
  batch_kwd = getattr(data_kwargs, 'batch_kwd', 'batch_sizes')
  batch_kwargs = {batch_kwd: batch_dims}

  dataset = data_loader(
      is_training=is_training,
      transpose=transpose,
      **batch_kwargs,
      **data_kwargs['load_kwargs'])

  is_numpy = getattr(data_kwargs, 'is_numpy', False)
  if not is_numpy:
    dataset = iter(tfds.as_numpy(dataset))

  return dataset


def resize(image: chex.Array, size: Tuple[int, int]) -> chex.Array:
  """Resizes a batch of images using bilinear interpolation."""
  return jax.image.resize(image,
                          (image.shape[0], size[0], size[1], image.shape[3]),
                          method='bilinear', antialias=False)
