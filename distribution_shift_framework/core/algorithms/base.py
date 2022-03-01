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

"""Base class for learning algorithms."""
import abc
from typing import Callable, Optional, Tuple

import chex
from distribution_shift_framework.core.datasets import data_utils
import haiku as hk


LossFn = Callable[..., chex.Array]


class LearningAlgorithm(hk.Module):
  """Class to encapsulate a learning algorithm."""

  def __init__(self, loss_fn: LossFn, name: str = 'DANN', **kwargs):
    """Initializes the algorithm with the given loss function."""
    super().__init__(name=name)
    self.loss_fn = loss_fn

  @abc.abstractmethod
  def __call__(
      self,
      logits: chex.Array,
      targets: chex.Array,
      reduction: str = 'mean',
      property_vs: Optional[chex.Array] = None
  ) -> Tuple[data_utils.ScalarDict, chex.Array]:
    """The loss function of the learning algorithm.

    Args:
      logits: The predicted logits input to the training algorithm.
      targets: The ground truth value to estimate.
      reduction: How to combine the loss for different samples.
      property_vs: An optional set of properties of the input data.

    Returns:
      scalars: A dictionary of key and scalar estimates. The key `loss`
        is the loss that should be minimized.
      preds: The raw softmax predictions.
    """
    pass

  def adversary(self,
                logits: chex.Array,
                property_vs: chex.Array,
                reduction: str = 'mean',
                targets: Optional[chex.Array] = None) -> data_utils.ScalarDict:
    """The adversarial loss function.

    If la = LearningAlgorithm(), this function is applied in a min-max game
    with la(). The model is trained to minimize the loss arising from la(),
    while maximizing the loss from the adversary (la.adversary()). The
    adversarial part of the model tries to minimize this loss.

    Args:
      logits: The predicted value input to the training algorithm.
      property_vs: An set of properties of the input data.
      reduction: How to combine the loss for different samples.
      targets: The ground truth value to estimate (optional).

    Returns:
      scalars: A dictionary of key and scalar estimates. The key `adv_loss` is
        the value that should be minimized (for the adversary) and maximized (
        for the model). If empty, this learning algorithm has no adversary.
    """
    # Do nothing.
    return {}
