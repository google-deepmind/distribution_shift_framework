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

"""Functions to create and combine hyper parameter sweeps."""

import functools
import itertools
from typing import Any, Dict, Iterable, List, Sequence

# A sweep is a list of parameter mappings that defines a set of experiments.
Sweep = List[Dict[str, Any]]


def sweep(parameter_name: str, values: Iterable[Any]) -> Sweep:
  """Creates a sweep from a list of values for a parameter."""
  return [{parameter_name: value} for value in values]


def product(sweeps: Sequence[Sweep]) -> Sweep:
  """Builds a sweep from the cartesian product of a list of sweeps."""
  return [functools.reduce(_combine_parameter_dicts, param_dicts, {})
          for param_dicts in itertools.product(*sweeps)]


def zipit(sweeps: Sequence[Sweep]) -> Sweep:
  """Builds a sweep from zipping a list of sweeps."""
  return [functools.reduce(_combine_parameter_dicts, param_dicts, {})
          for param_dicts in zip(*sweeps)]


def _combine_parameter_dicts(x: Dict[str, Any], y: Dict[str, Any]
                             ) -> Dict[str, Any]:
  if x.keys() & y.keys():
    raise ValueError('Cannot combine sweeps that set the same parameters. '
                     f'Keys in x: {x.keys()}, keys in y: {y.keys}, '
                     f'overlap: {x.keys() & y.keys()}')
  return {**x, **y}

