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

"""Tests for distribution_shift_framework.core.checkpointing."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from distribution_shift_framework.core import checkpointing
import jax
import ml_collections
import numpy as np
import numpy.testing as npt
import tensorflow as tf


class CheckpointingTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(data={}),
      dict(data={'params': []}),
      dict(data={'state': None}),
      dict(data={'params': 3, 'stuff': 5.3, 'something': 'anything'}),
      dict(data={'params': {'stuff': 5.3, 'something': 'anything'}}),
      dict(data={'params': {'stuff': {'something': 'anything'}}}),
      dict(data={'params': {'stuff': {'something': np.random.rand(4, 3, 2)}}}),
  ])
  def test_load_and_save_model(self, data):
    ckpt_file = os.path.join(self.create_tempdir(), 'ckpt.pkl')
    checkpointing.save_model(ckpt_file, data)
    loaded_data = checkpointing.load_model(ckpt_file)
    loaded_leaves, loaded_treedef = jax.tree_flatten(loaded_data)
    leaves, treedef = jax.tree_flatten(data)
    for leaf, loaded_leaf in zip(leaves, loaded_leaves):
      npt.assert_array_equal(leaf, loaded_leaf)
    self.assertEqual(treedef, loaded_treedef)

  def test_empty_checkpoint_dir(self):
    config = ml_collections.ConfigDict()
    config.checkpoint_dir = None
    self.assertIsNone(checkpointing.get_checkpoint_dir(config))

  def test_get_checkpoint_dir(self):
    config = ml_collections.ConfigDict()
    temp_dir = self.create_tempdir()
    config.checkpoint_dir = os.path.join(temp_dir, 'my_exp')
    self.assertFalse(tf.io.gfile.exists(config.checkpoint_dir))
    config.host_subdir = 'prefix_{host_id}_postfix'
    path = checkpointing.get_checkpoint_dir(config)
    self.assertEqual(os.path.join(temp_dir, 'my_exp', 'prefix_0_postfix'), path)
    self.assertTrue(tf.io.gfile.exists(path))


if __name__ == '__main__':
  absltest.main()
