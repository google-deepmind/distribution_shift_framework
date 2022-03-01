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

"""Run the formalisation pipeline."""

import functools

from absl import app
from absl import flags
from distribution_shift_framework.classification import experiment_lib
from jaxline import platform


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, experiment_lib.Experiment))

