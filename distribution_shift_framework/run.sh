#!/bin/sh
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euf -o pipefail  # Stop at failure.

python3 -m venv /tmp/distribution_shift_framework
source /tmp/distribution_shift_framework/bin/activate
pip install -U pip
pip install -r distribution_shift_framework/requirements.txt

python3 -m distribution_shift_framework.classification.experiment_lib_test
