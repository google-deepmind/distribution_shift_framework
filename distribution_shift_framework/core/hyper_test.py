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

"""Tests for distribution_shift_framework.core.hyper."""


from absl.testing import absltest
from absl.testing import parameterized
from distribution_shift_framework.core import hyper


class HyperTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(parameter_name='a', values=[1, 2, 3],
           expected_sweep=[{'a': 1}, {'a': 2}, {'a': 3}]),
      dict(parameter_name='b', values=[.1, .2, .3],
           expected_sweep=[{'b': .1}, {'b': .2}, {'b': .3}]),
      dict(parameter_name='c', values=[True, False],
           expected_sweep=[{'c': True}, {'c': False}]),
      dict(parameter_name='d', values=['one', 'two', 'three'],
           expected_sweep=[{'d': 'one'}, {'d': 'two'}, {'d': 'three'}]),
      dict(parameter_name='e', values=[1, 0.5, True, 'string'],
           expected_sweep=[{'e': 1}, {'e': 0.5}, {'e': True}, {'e': 'string'}]),
      dict(parameter_name='f', values=[],
           expected_sweep=[]),
  ])
  def test_sweep(self, parameter_name, values, expected_sweep):
    self.assertEqual(expected_sweep, hyper.sweep(parameter_name, values))

  @parameterized.parameters([
      dict(sweeps=[],
           expected_sweep=[{}]),
      dict(sweeps=[hyper.sweep('param1', [1, 2, 3, 4, 5, 6])],
           expected_sweep=[
               {'param1': 1}, {'param1': 2}, {'param1': 3},
               {'param1': 4}, {'param1': 5}, {'param1': 6},
           ]),
      dict(sweeps=[hyper.sweep('param1', [1, 2, 3]),
                   hyper.sweep('param2', [4, 5, 6])],
           expected_sweep=[
               {'param1': 1, 'param2': 4},
               {'param1': 1, 'param2': 5},
               {'param1': 1, 'param2': 6},
               {'param1': 2, 'param2': 4},
               {'param1': 2, 'param2': 5},
               {'param1': 2, 'param2': 6},
               {'param1': 3, 'param2': 4},
               {'param1': 3, 'param2': 5},
               {'param1': 3, 'param2': 6},
           ]),
      dict(sweeps=[hyper.sweep('param1', [1, 2]),
                   hyper.sweep('param2', [3, 4]),
                   hyper.sweep('param3', [5, 6])],
           expected_sweep=[
               {'param1': 1, 'param2': 3, 'param3': 5},
               {'param1': 1, 'param2': 3, 'param3': 6},
               {'param1': 1, 'param2': 4, 'param3': 5},
               {'param1': 1, 'param2': 4, 'param3': 6},
               {'param1': 2, 'param2': 3, 'param3': 5},
               {'param1': 2, 'param2': 3, 'param3': 6},
               {'param1': 2, 'param2': 4, 'param3': 5},
               {'param1': 2, 'param2': 4, 'param3': 6},
           ]),
      dict(sweeps=[hyper.sweep('param1', [1, 2., 'Three']),
                   hyper.sweep('param2', [True, 'Two', 3.0])],
           expected_sweep=[
               {'param1': 1, 'param2': True},
               {'param1': 1, 'param2': 'Two'},
               {'param1': 1, 'param2': 3.0},
               {'param1': 2., 'param2': True},
               {'param1': 2., 'param2': 'Two'},
               {'param1': 2., 'param2': 3.0},
               {'param1': 'Three', 'param2': True},
               {'param1': 'Three', 'param2': 'Two'},
               {'param1': 'Three', 'param2': 3.0},
           ]),
  ])
  def test_product(self, sweeps, expected_sweep):
    self.assertEqual(expected_sweep, hyper.product(sweeps))

  def test_product_raises_valueerror_for_same_name(self):
    sweep1 = hyper.sweep('param1', [1, 2, 3])
    sweep2 = hyper.sweep('param2', [4, 5, 6])
    sweep3 = hyper.sweep('param1', [7, 8, 9])
    with self.assertRaises(ValueError):
      hyper.product([sweep1, sweep2, sweep3])

  @parameterized.parameters([
      dict(sweeps=[],
           expected_sweep=[]),
      dict(sweeps=[hyper.sweep('param1', [1, 2, 3, 4, 5, 6])],
           expected_sweep=[
               {'param1': 1}, {'param1': 2}, {'param1': 3},
               {'param1': 4}, {'param1': 5}, {'param1': 6},
           ]),
      dict(sweeps=[hyper.sweep('param1', [1, 2, 3]),
                   hyper.sweep('param2', [4, 5, 6])],
           expected_sweep=[
               {'param1': 1, 'param2': 4},
               {'param1': 2, 'param2': 5},
               {'param1': 3, 'param2': 6},
           ]),
      dict(sweeps=[hyper.sweep('param1', [1, 2, 3]),
                   hyper.sweep('param2', [4, 5, 6]),
                   hyper.sweep('param3', [7, 8, 9])],
           expected_sweep=[
               {'param1': 1, 'param2': 4, 'param3': 7},
               {'param1': 2, 'param2': 5, 'param3': 8},
               {'param1': 3, 'param2': 6, 'param3': 9},
           ]),
      dict(sweeps=[hyper.sweep('param1', [1, 2., 'Three']),
                   hyper.sweep('param2', [True, 'Two', 3.0])],
           expected_sweep=[
               {'param1': 1, 'param2': True},
               {'param1': 2., 'param2': 'Two'},
               {'param1': 'Three', 'param2': 3.0},
           ]),
      dict(sweeps=[hyper.sweep('param1', [1, 2, 3]),
                   hyper.sweep('param2', [4, 5, 6, 7])],
           expected_sweep=[
               {'param1': 1, 'param2': 4},
               {'param1': 2, 'param2': 5},
               {'param1': 3, 'param2': 6},
           ]),
  ])
  def test_zipit(self, sweeps, expected_sweep):
    self.assertEqual(expected_sweep, hyper.zipit(sweeps))

  def test_zipit_raises_valueerror_for_same_name(self):
    sweep1 = hyper.sweep('param1', [1, 2, 3])
    sweep2 = hyper.sweep('param2', [4, 5, 6])
    sweep3 = hyper.sweep('param1', [7, 8, 9])
    with self.assertRaises(ValueError):
      hyper.zipit([sweep1, sweep2, sweep3])


if __name__ == '__main__':
  absltest.main()
