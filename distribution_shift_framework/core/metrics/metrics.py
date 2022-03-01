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

"""List of standard evaluation metrics."""
from typing import Dict, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
import sklearn
import sklearn.metrics


def pearson(predictions: chex.Array, labels: chex.Array) -> chex.Scalar:
  """Computes the Pearson correlation coefficient.

  Assumes all inputs are numpy arrays.

  Args:
    predictions: The predicted class labels.
    labels: The true class labels.

  Returns:
    cc: The predicted Pearson correlation coefficient.
  """
  cc = stats.pearsonr(predictions, labels)[0]
  return cc


def f1_score(average: chex.Array,
             predictions: chex.Array,
             labels: chex.Array) -> chex.Scalar:
  """Computes the F1 score.

  Assumes all inputs are numpy arrays.

  Args:
    average: How to accumulate the f1 score (macro or weighted).
    predictions: The predicted class labels.
    labels: The true class labels.

  Returns:
    f1: The predicted f1 score.
  """
  f1 = sklearn.metrics.f1_score(
      predictions, labels, average=average, labels=np.unique(labels))

  return f1


def recall_score(average: chex.Array,
                 predictions: chex.Array,
                 labels: chex.Array) -> chex.Scalar:
  """Computes the recall score.

  Assumes all inputs are numpy arrays.

  Args:
    average: How to accumulate the recall score (macro or weighted).
    predictions: The predicted class labels.
    labels: The true class labels.

  Returns:
    recall: The predicted recall.
  """
  recall = sklearn.metrics.recall_score(
      predictions, labels, average=average, labels=np.unique(labels))

  return recall


def top_k_accuracy(logits: chex.Array,
                   labels: chex.Array,
                   k: int) -> chex.Scalar:
  """Compute top_k_accuracy.

  Args:
    logits: The network predictions.
    labels: The true class labels.
    k: Accuracy at what k.

  Returns:
    top_k_accuracy: The top k accuracy.
  """
  chex.assert_equal_shape_prefix([logits, labels], 1)
  chex.assert_rank(logits, 2)  # [bs, k]
  chex.assert_rank(labels, 1)  # [bs]

  _, top_ks = jax.vmap(lambda x: jax.lax.top_k(x, k=k))(logits)

  return jnp.mean(jnp.sum(top_ks == labels[:, None], axis=-1))


def compute_all_metrics(predictions: chex.Array, labels: chex.Array,
                        metrics: Sequence[str]) -> Dict[str, chex.Scalar]:
  """Computes a set of metrics given the predictions and labels.

  Args:
    predictions: A tensor of shape (N, *): the predicted values.
    labels: A tensor of shape (N, *): the ground truth values.
    metrics: A sequence of strings describing the metrics to be evaluated.
      This can be one of 'pearson' (to compute the pearson correlation
      coefficient), 'f1_{average}', 'recall_{average}'. For f1 and
      recall the value {average} is defined in the numpy api:
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

  Returns:
    scalars: A dict containing (metric name, score) items with the metric name
    and associated score as a float value.
  """
  scalars = {}
  for metric in metrics:
    if metric == 'pearson':
      scalars['pearson'] = pearson(predictions, labels)
    elif 'f1' in metric:
      scalars[metric] = f1_score(metric.split('_')[1], predictions, labels)
    elif 'recall' in metric:
      scalars[metric] = recall_score(metric.split('_')[1], predictions, labels)

  return scalars


def top1_accuracy(labels, features, predictions, latents):
  del features
  del latents
  return np.equal(predictions, labels).mean()
