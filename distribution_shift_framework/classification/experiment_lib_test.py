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

"""Tests for distribution_shift_framework.classification.experiment_lib."""

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from distribution_shift_framework.classification import config
from distribution_shift_framework.classification import experiment_lib
import jax
from jaxline import platform


_PREV_JAX_CONFIG = None


def setUpModule():
  global _PREV_JAX_CONFIG
  _PREV_JAX_CONFIG = jax.config.values.copy()
  # Disable jax optimizations to speed up test.
  jax.config.update('jax_disable_most_optimizations', True)


def tearDownModule():
  # Set config to previous values.
  jax.config.values.update(**_PREV_JAX_CONFIG)


class ExperimentLibTest(parameterized.TestCase):

  @parameterized.parameters([
      # Different algorithms.
      dict(algorithm='CORAL', test_case='ood', model='resnet18',
           dataset_name='dsprites', label='label_shape',
           property_label='label_color', number_of_seeds=1),
      dict(algorithm='DANN', test_case='ood', model='resnet18',
           dataset_name='dsprites', label='label_shape',
           property_label='label_color', number_of_seeds=1),
      dict(algorithm='ERM', test_case='ood', model='resnet18',
           dataset_name='dsprites', label='label_shape',
           property_label='label_color', number_of_seeds=1),
      dict(algorithm='IRM', test_case='ood', model='resnet18',
           dataset_name='dsprites', label='label_shape',
           property_label='label_color', number_of_seeds=1),
      dict(algorithm='SagNet', test_case='ood', model='resnet18',
           dataset_name='dsprites', label='label_shape',
           property_label='label_color', number_of_seeds=1),
      # Different datasets.
      dict(algorithm='ERM', test_case='ood', model='resnet18',
           dataset_name='small_norb', label='label_category',
           property_label='label_azimuth', number_of_seeds=1),
      dict(algorithm='ERM', test_case='ood', model='resnet18',
           dataset_name='shapes3d', label='label_shape',
           property_label='label_object_hue', number_of_seeds=1),
      # Different test cases.
      dict(algorithm='ERM', test_case='lowdata', model='resnet18',
           dataset_name='shapes3d', label='label_shape',
           property_label='label_object_hue', number_of_seeds=1),
      dict(algorithm='ERM', test_case='correlated.lowdata', model='resnet18',
           dataset_name='shapes3d', label='label_shape',
           property_label='label_object_hue', number_of_seeds=1),
      dict(algorithm='ERM', test_case='lowdata.noise', model='resnet18',
           dataset_name='shapes3d', label='label_shape',
           property_label='label_object_hue', number_of_seeds=1),
      dict(algorithm='ERM', test_case='lowdata.fixeddata', model='resnet18',
           dataset_name='shapes3d', label='label_shape',
           property_label='label_object_hue', number_of_seeds=1),
  ])
  def test_train(self, **kwargs):
    kwargs['training_steps'] = 3
    kwargs['use_fake_data'] = True
    kwargs['batch_size'] = 8
    options = ','.join([f'{k}={v}' for k, v in kwargs.items()])
    cfg = config.get_config(options)
    with flagsaver.flagsaver(config=cfg, jaxline_mode='train'):
      platform.main(experiment_lib.Experiment, [])


if __name__ == '__main__':
  absltest.main()
