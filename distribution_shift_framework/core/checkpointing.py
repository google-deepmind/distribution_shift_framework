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

"""Checkpointing code.
"""
import os
import pickle
from typing import Mapping, Optional

import jax
import ml_collections
import optax
import tensorflow as tf


def load_model(checkpoint_path: str) -> Mapping[str, optax.Params]:
  with tf.io.gfile.GFile(checkpoint_path, 'rb') as f:
    return pickle.load(f)


def save_model(checkpoint_path: str,
               ckpt_dict: Mapping[str, optax.Params]):
  with tf.io.gfile.GFile(checkpoint_path, 'wb') as f:
    # Using protocol 4 as it's the default from Python 3.8 on.
    pickle.dump(ckpt_dict, f, protocol=4)


def get_checkpoint_dir(config: ml_collections.ConfigDict) -> Optional[str]:
  """Constructs the checkpoint directory from the config."""

  if config.checkpoint_dir is None:
    return None
  path = os.path.join(config.checkpoint_dir,
                      config.host_subdir.format(host_id=jax.process_index()))
  if not tf.io.gfile.exists(path):
    tf.io.gfile.makedirs(path)
  return path
