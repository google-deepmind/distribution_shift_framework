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

import contextlib
import functools
import os
from typing import Generator, Mapping, Optional, Tuple

from absl import flags
from absl import logging
import chex
from distribution_shift_framework.core import checkpointing
from distribution_shift_framework.core.datasets import data_utils
from distribution_shift_framework.core.metrics import metrics
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import utils
import ml_collections
import numpy as np
import optax
from six.moves import cPickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


def get_per_device_batch_size(total_batch_size: int) -> int:
  num_devices = jax.device_count()
  per_device_batch_size, ragged = divmod(total_batch_size, num_devices)

  if ragged:
    raise ValueError(
        f'Global batch size {total_batch_size} must be divisible by the '
        f'total number of devices {num_devices}')
  return per_device_batch_size


class Experiment(experiment.AbstractExperiment):
  """Formalisation experiment."""
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_opt_state': 'opt_state',
      '_d_params': 'd_params',
      '_d_state': 'd_state',
      '_d_opt_state': 'd_opt_state',
      '_adapt_params': 'adapt_params',
      '_adapt_state': 'adapt_state'
  }

  def __init__(self, mode: str, init_rng: chex.PRNGKey,
               config: ml_collections.ConfigDict):
    """Initializes experiment."""

    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)

    self.mode = mode
    self.config = config
    self.init_rng = init_rng

    # Set up discriminator parameters.
    self._d_params = None
    self._d_state = None
    self._d_opt_state = None

    # Double transpose trick to improve performance on TPUs.
    self._should_transpose_images = (
        config.enable_double_transpose and
        jax.local_devices()[0].platform == 'tpu')

    self._params = None  # params
    self._state = None  # network state for stats like batchnorm
    self._opt_state = None  # optimizer state
    self._adapt_params = None
    self._adapt_state = None
    self._label = config.data.label

    with utils.log_activity('transform functions'):
      self.forward = hk.transform_with_state(self._forward_fn)
      self.eval_batch = jax.pmap(self._eval_batch, axis_name='i')
      self.learner_fn = hk.transform_with_state(self._learner_fn)
      self.adversarial_fn = hk.transform_with_state(self._adversarial_fn)
      self.adapt_fn = self._adapt_fn
      self.adaptor = None

      self._update_func = jax.pmap(
          self._update_func, axis_name='i', donate_argnums=(0, 1, 2))

    if mode == 'train':
      with utils.log_activity('initialize training'):
        self._init_train(init_rng)

      if getattr(self.config.training.learn_adapt, 'fn', None):
        learner_adapt_fn = self.config.training.learn_adapt.fn
        learner_adapt_kwargs = self.config.training.learn_adapt.kwargs
        self._train_adapter = learner_adapt_fn(**learner_adapt_kwargs)
        if self._adapt_params is None:
          self._adapt_params = self._params
          self._adapt_state = self._state
        self._train_adapter.set(self._adapt_params, self._adapt_state)
      else:
        self._train_adapter = None

  def optimizer(self) -> optax.GradientTransformation:
    optimizer_fn = getattr(optax, self.config.optimizer.name)
    return optimizer_fn(**self.config.optimizer.kwargs)

  def _maybe_undo_transpose_images(self, images: chex.Array) -> chex.Array:
    if self._should_transpose_images:
      return jnp.transpose(images, (1, 2, 3, 0))  # NHWC -> HWCN.
    return images

  def _maybe_transpose_images(self, images: chex.Array) -> chex.Array:
    if self._should_transpose_images:
      # We use the double transpose trick to improve performance for TPUs.
      # Note that there is a matching NHWC->HWCN transpose in the data pipeline.
      # Here we reset back to NHWC like our model expects. The compiler cannot
      # make this optimization for us since our data pipeline and model are
      # compiled separately.
      images = jnp.transpose(images, (3, 0, 1, 2))  # HWCN -> NHWC.
    return images

  def _postprocess_fn(
      self,
      inputs: data_utils.Batch,
      rng: chex.PRNGKey
  ) -> data_utils.Batch:
    if not hasattr(self.config, 'postprocessing'):
      return inputs
    postprocessing = getattr(self.config.postprocessing, 'fn', None)
    if postprocessing is None:
      return inputs
    postprocess_fn = functools.partial(postprocessing,
                                       **self.config.postprocessing.kwargs)
    images = inputs['image']
    labels = inputs['one_hot_label']
    postprocessed_images, postprocessed_labels = postprocess_fn(
        images, labels, rng=rng)

    postprocessed_inputs = dict(**inputs)
    postprocessed_inputs['image'] = postprocessed_images
    postprocessed_inputs['one_hot_label'] = postprocessed_labels
    return postprocessed_inputs

  def _learner_fn(self, inputs: data_utils.Batch,
                  reduction='mean') -> Tuple[data_utils.ScalarDict, chex.Array]:

    logits = self._forward_fn(inputs, is_training=True)

    if getattr(self.config.data, 'label_property', '') in inputs.keys():
      property_vs = inputs[self.config.data.label_property]
      property_onehot = hk.one_hot(property_vs, self.config.data.n_properties)
    else:
      property_onehot = None

    algorithm_fn = self.config.training.algorithm.fn
    kwargs = self.config.training.algorithm.kwargs
    scalars, logits = algorithm_fn(**kwargs)(
        logits, inputs['one_hot_label'], property_vs=property_onehot,
        reduction=reduction)

    predicted_label = jnp.argmax(logits, axis=-1)
    top1_acc = jnp.equal(predicted_label,
                         inputs[self._label]).astype(jnp.float32)
    scalars['top1_acc'] = top1_acc.mean()

    return scalars, logits

  def learner_adapt_weights_fn(
      self, params: optax.Params, state: optax.OptState,
      old_params: optax.Params, old_state: optax.OptState,
      inputs: data_utils.Batch, rng: chex.PRNGKey,
      global_step: chex.Array
      ) -> Tuple[Tuple[data_utils.ScalarDict, chex.Array], optax.OptState]:
    (scalars, logits), g_state = self._train_adapter(
        fn=functools.partial(self.learner_fn.apply, reduction=None),
        params=params, state=state, inputs=inputs, global_step=global_step,
        rng=rng, old_params=old_params, old_state=old_state)
    return (scalars, logits), g_state

  def _adversarial_fn(self, logits: chex.Array,
                      inputs: data_utils.Batch) -> data_utils.ScalarDict:
    if getattr(self.config.data, 'label_property', '') in inputs.keys():
      property_vs = inputs[self.config.data.label_property]
      property_onehot = hk.one_hot(property_vs, self.config.data.n_properties)
    else:
      property_onehot = None

    one_hot_labels = inputs['one_hot_label']
    algorithm_fn = self.config.training.algorithm.fn
    kwargs = self.config.training.algorithm.kwargs
    return algorithm_fn(**kwargs).adversary(
        logits, property_vs=property_onehot, reduction='mean',
        targets=one_hot_labels)

  def _adapt_fn(self, params: optax.Params, state: optax.OptState,
                rng: chex.PRNGKey, is_final_eval: bool = False):
    adapt_fn = getattr(self.config.adapter, 'fn')
    adapt_kwargs = getattr(self.config.adapter, 'kwargs')

    forward_fn = functools.partial(self.forward.apply, is_training=True,
                                   test_local_stats=False)
    self.adaptor = adapt_fn(init_params=params,
                            init_state=state,
                            forward=jax.pmap(forward_fn, axis_name='i'),
                            **adapt_kwargs)

    per_device_batch_size = get_per_device_batch_size(
        self.config.training.batch_size)

    ds = self._load_data(per_device_batch_size=per_device_batch_size,
                         is_training=False,
                         data_kwargs=self.config.data.test_kwargs)

    for step, batch in enumerate(ds, 1):
      logging.info('Updating using an adaptor function.')
      self.adaptor.update(batch, batch[self.config.data.label_property], rng)
      if (not is_final_eval and
          step > getattr(self.config.adapter, 'num_adaptation_steps')):
        break

  def _forward_fn(self,
                  inputs: data_utils.Batch,
                  is_training: bool,
                  test_local_stats: bool = False) -> chex.Array:
    model_constructor = self.config.model.constructor
    model_instance = model_constructor(**self.config.model.kwargs.to_dict())

    images = inputs['image']
    images = self._maybe_transpose_images(images)
    images = self.config.model.preprocess(images)

    if isinstance(model_instance, hk.nets.MLP):
      return model_instance(images)
    return model_instance(images, is_training=is_training)

  def _d_loss_fn(
      self, d_params: optax.Params, d_state: optax.OptState, inputs: chex.Array,
      logits: chex.Array,
      rng: chex.PRNGKey
      ) -> Tuple[chex.Array, Tuple[data_utils.ScalarDict, optax.OptState]]:

    d_scalars, d_state = self.adversarial_fn.apply(d_params, d_state, rng,
                                                   logits, inputs)
    if not d_scalars:
      # No adversary.
      return 0., (d_scalars, d_state)

    scaled_loss = d_scalars['loss'] / jax.device_count()
    d_scalars = {f'adv_{k}': v for k, v in d_scalars.items()}

    return scaled_loss, (d_scalars, d_state)

  def _run_postprocess_fn(self,
                          rng: chex.PRNGKey,
                          inputs: data_utils.Batch) -> data_utils.Batch:
    inputs = self._postprocess_fn(inputs, rng)
    return inputs

  def _loss_fn(
      self, g_params: optax.Params,
      g_state: optax.OptState,
      d_params: optax.Params,
      d_state: optax.OptState,
      inputs: chex.Array,
      rng: chex.PRNGKey,
      global_step: chex.Array,
      old_g_params: Optional[optax.Params] = None,
      old_g_state: Optional[optax.OptState] = None
  ) -> Tuple[chex.Array, Tuple[
      data_utils.ScalarDict, chex.Array, data_utils.Batch, optax.OptState]]:
    # Find the loss according to the generator.
    if getattr(self.config.training.learn_adapt, 'fn', None):
      # Use generator loss computed by a training adaptation algorithm.
      (scalars, logits), g_state = self.learner_adapt_weights_fn(
          params=g_params,
          state=g_state,
          old_params=old_g_params,
          old_state=old_g_state,
          rng=rng,
          inputs=inputs,
          global_step=global_step)
    else:
      (scalars, logits), g_state = self.learner_fn.apply(g_params, g_state, rng,
                                                         inputs)

    d_scalars, _ = self.adversarial_fn.apply(d_params, d_state, rng, logits,
                                             inputs)

    # If there is an adversary:
    if 'loss' in d_scalars.keys():
      # Want to minimize the loss, so negate it.
      adv_weight = self.config.training.adversarial_weight
      scalars['loss'] = scalars['loss'] - d_scalars['loss'] * adv_weight
      scalars.update({f'gen_adv_{k}': v for k, v in d_scalars.items()})

    scaled_loss = scalars['loss'] / jax.device_count()
    return scaled_loss, (scalars, logits, inputs, g_state)

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def _prepare_train_batch(self, rng: chex.PRNGKey,
                           batch: data_utils.Batch) -> data_utils.Batch:
    noise_threshold = self.config.training.label_noise
    if noise_threshold > 0:
      random_labels = jax.random.randint(
          rng[0],
          shape=batch[self._label].shape,
          dtype=batch[self._label].dtype,
          minval=0,
          maxval=self.config.data.n_classes)
      mask = jax.random.uniform(rng[0],
                                batch[self._label].shape) < noise_threshold
      batch[self._label] = (random_labels * mask +
                            batch[self._label] * (1 - mask))
    batch['one_hot_label'] = hk.one_hot(
        batch[self._label], self.config.data.n_classes)
    return batch

  def _init_train(self, rng: chex.PRNGKey):
    self._train_input = utils.py_prefetch(self._build_train_input)

    if self._params is None:
      logging.info('Initializing parameters randomly rather than restoring'
                   ' from checkpoint.')
      batch = next(self._train_input)
      batch['one_hot_label'] = hk.one_hot(batch[self._label],
                                          self.config.data.n_classes)

      # Initialize generator.
      self._params, self._state = self._init_params(rng, batch)
      opt_init, _ = self.optimizer()
      self._opt_state = jax.pmap(opt_init)(self._params)

      # Initialize discriminator.
      bcast_rng = utils.bcast_local_devices(rng)
      (_, dummy_logits), _ = jax.pmap(self.learner_fn.apply)(self._params,
                                                             self._state,
                                                             bcast_rng, batch)
      self._d_params, self._d_state = self._init_d_params(
          rng, dummy_logits, batch)
      opt_init, _ = self.optimizer()
      if self._d_params:
        self._d_opt_state = jax.pmap(opt_init)(self._d_params)
      else:
        # Is empty.
        self._d_opt_state = None

  def _init_params(
      self, rng: chex.PRNGKey,
      batch: data_utils.Batch) -> Tuple[optax.Params, optax.OptState]:
    init_net = jax.pmap(self.learner_fn.init)
    rng = utils.bcast_local_devices(rng)
    params, state = init_net(rng, batch)
    if not self.config.pretrained_checkpoint:
      return params, state
    ckpt_data = checkpointing.load_model(
        self.config.pretrained_checkpoint)
    ckpt_params, ckpt_state = ckpt_data['params'], ckpt_data['state']

    ckpt_params = utils.bcast_local_devices(ckpt_params)
    ckpt_state = utils.bcast_local_devices(ckpt_state)

    def use_pretrained_if_shapes_match(params, ckpt_params):
      if params.shape == ckpt_params.shape:
        return ckpt_params
      logging.warning('Shape mismatch! Initialized parameter: %s, '
                      'Pretrained parameter: %s.',
                      params.shape, ckpt_params.shape)
      return params

    params = jax.tree_multimap(
        use_pretrained_if_shapes_match, params, ckpt_params)
    return params, ckpt_state

  def _init_d_params(
      self, rng: chex.PRNGKey, logits: chex.Array,
      batch: data_utils.Batch) -> Tuple[optax.Params, optax.OptState]:
    init_net = jax.pmap(self.adversarial_fn.init)
    rng = utils.bcast_local_devices(rng)
    return init_net(rng, logits, batch)

  def _write_images(self, writer, global_step: chex.Array,
                    images: Mapping[str, chex.Array]):
    global_step = np.array(utils.get_first(global_step))

    images_to_write = {
        k: self._maybe_transpose_images(utils.get_first(v))
        for k, v in images.items()}

    writer.write_images(global_step, images_to_write)

  def _load_data(self,
                 per_device_batch_size: int,
                 is_training: bool,
                 data_kwargs: ml_collections.ConfigDict
                 ) -> Generator[data_utils.Batch, None, None]:

    with contextlib.ExitStack() as stack:
      if self.config.use_fake_data:
        stack.enter_context(tfds.testing.mock_data(num_examples=128))
      ds = data_utils.load_dataset(
          is_training=is_training,
          batch_dims=[jax.local_device_count(), per_device_batch_size],
          transpose=self._should_transpose_images,
          data_kwargs=data_kwargs)
    return ds

  def _build_train_input(self) -> Generator[data_utils.Batch, None, None]:
    per_device_batch_size = get_per_device_batch_size(
        self.config.training.batch_size)
    return self._load_data(per_device_batch_size=per_device_batch_size,
                           is_training=True,
                           data_kwargs=self.config.data.train_kwargs)

  def _update_func(
      self,
      params: optax.Params,
      state: optax.OptState,
      opt_state: optax.OptState,
      global_step: chex.Array,
      batch: data_utils.Batch,
      rng: chex.PRNGKey,
      old_g_params: Optional[optax.Params] = None,
      old_g_state: Optional[optax.OptState] = None
  ) -> Tuple[Tuple[optax.Params, optax.Params], Tuple[
      optax.OptState, optax.OptState], Tuple[optax.OptState, optax.OptState],
             data_utils.ScalarDict, data_utils.Batch]:
    """Updates parameters ."""
    # Obtain the parameters and discriminators.
    (g_params, d_params) = params
    (g_state, d_state) = state
    (g_opt_state, d_opt_state) = opt_state

    ################
    # Generator.
    ################
    # Compute the loss for the generator.
    inputs = self._run_postprocess_fn(rng, batch)
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    scaled_grads, (g_scalars, logits, preprocessed_inputs,
                   g_state) = grad_loss_fn(g_params, g_state, d_params, d_state,
                                           inputs, rng, global_step,
                                           old_g_params=old_g_params,
                                           old_g_state=old_g_state)

    # Update the generator.
    grads = jax.lax.psum(scaled_grads, axis_name='i')
    _, opt_apply = self.optimizer()

    updates, g_opt_state = opt_apply(grads, g_opt_state, g_params)
    g_params = optax.apply_updates(g_params, updates)

    ################
    # Discriminator.
    ################
    if not self._d_opt_state:
      # No discriminator.
      scalars = dict(global_step=global_step, **g_scalars)
      return ((g_params, d_params), (g_state, d_state),
              (g_opt_state, d_opt_state), scalars, preprocessed_inputs)

    # Compute the loss for the discriminator.
    grad_loss_fn = jax.grad(self._d_loss_fn, has_aux=True)
    scaled_grads, (d_scalars, d_state) = grad_loss_fn(d_params, d_state, batch,
                                                      logits, rng)

    # Update the discriminator.
    grads = jax.lax.psum(scaled_grads, axis_name='i')
    _, opt_apply = self.optimizer()

    updates, d_opt_state = opt_apply(grads, d_opt_state, d_params)
    d_params = optax.apply_updates(d_params, updates)

    # For logging while training.
    scalars = dict(
        global_step=global_step,
        **g_scalars,
        **d_scalars)
    return ((g_params, d_params), (g_state, d_state),
            (g_opt_state, d_opt_state), scalars, preprocessed_inputs)

  def step(self, global_step: chex.Array, rng: chex.PRNGKey, writer,
           **unused_kwargs) -> chex.Array:
    """Perform one step of the model."""

    batch = next(self._train_input)
    batch = self._prepare_train_batch(rng, batch)

    params, state, opt_state, scalars, preprocessed_batch = (
        self._update_func(
            params=(self._params, self._d_params),
            state=(self._state, self._d_state),
            opt_state=(self._opt_state, self._d_opt_state),
            global_step=global_step,
            batch=batch,
            rng=rng,
            old_g_params=self._adapt_params,
            old_g_state=self._adapt_state))
    (self._params, self._d_params) = params
    (self._state, self._d_state) = state
    (self._opt_state, self._d_opt_state) = opt_state

    if self._train_adapter:
      self._adapt_params, self._adapt_state = self._train_adapter.update(
          self._params, self._state, utils.get_first(global_step))

    images = batch['image']
    preprocessed_images = preprocessed_batch['image']

    if self.config.training.save_images:
      self._write_images(writer, global_step,
                         {'images': images,
                          'preprocessed_images': preprocessed_images})

    # Just return the tracking metrics on the first device for logging.
    return utils.get_first(scalars)

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def _load_eval_data(
      self,
      per_device_batch_size: int) -> Generator[data_utils.Batch, None, None]:
    return self._load_data(per_device_batch_size=per_device_batch_size,
                           is_training=False,
                           data_kwargs=self.config.data.test_kwargs)

  def _full_eval(self, rng: chex.PRNGKey, scalars: data_utils.ScalarDict,
                 checkpoint_path: Optional[str] = None
                 ) -> data_utils.ScalarDict:
    if checkpoint_path:
      ckpt_data = checkpointing.load_model(checkpoint_path)
      params, state = ckpt_data['params'], ckpt_data['state']
      params = utils.bcast_local_devices(params)
      state = utils.bcast_local_devices(state)
    else:
      params, state = self._params, self._state

    # Iterate over all the test sets.
    original_subset = self.config.data.test_kwargs.load_kwargs.subset
    for test_subset in getattr(self.config.data, 'test_sets', ('test',)):
      self.config.data.test_kwargs.load_kwargs.subset = test_subset
      test_scalars = jax.device_get(
          self._eval_top1_accuracy(params, state, rng, is_final=True))
      scalars.update(
          {f'{test_subset}_{k}': v for k, v in test_scalars.items()})
    self.config.data.test_kwargs.load_kwargs.subset = original_subset
    return scalars

  def evaluate(self, global_step: chex.Array, rng: chex.PRNGKey, writer,
               **unused_args) -> data_utils.ScalarDict:
    """See base class."""
    # Need to set these so `on_new_best_model` can do a full eval.
    self._writer = writer
    self._rng = rng
    global_step = np.array(utils.get_first(global_step))

    scalars = jax.device_get(
        self._eval_top1_accuracy(self._params, self._state, rng))

    if FLAGS.config.eval_specific_checkpoint_dir:
      scalars = self._full_eval(rng, scalars,
                                FLAGS.config.eval_specific_checkpoint_dir)

    logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars

  def on_new_best_model(self, best_state: ml_collections.ConfigDict):
    scalars = self._full_eval(self._rng, {})
    if self._writer is not None:
      self._writer.write_scalars(best_state.global_step, scalars)
    ckpt_data = {}
    for self_key, ckpt_key in self.CHECKPOINT_ATTRS.items():
      ckpt_data[ckpt_key] = getattr(self, self_key)
    checkpoint_path = checkpointing.get_checkpoint_dir(FLAGS.config)
    checkpointing.save_model(os.path.join(checkpoint_path, 'best.pkl'),
                             ckpt_data)

  def _eval_top1_accuracy(self, params: optax.Params, state: optax.OptState,
                          rng: chex.PRNGKey, is_final: bool = False
                          ) -> data_utils.ScalarDict:
    """Evaluates an epoch."""
    total_batch_size = self.config.evaluation.batch_size
    per_device_batch_size = total_batch_size
    eval_data = self._load_eval_data(per_device_batch_size)

    # If using an adaptive method.
    if getattr(self.config.adapter, 'fn', None):
      self.adapt_fn(params, state, rng, is_final_eval=is_final)
      self.adaptor.set_up_eval()

    # Accuracies for each set of corruptions.
    labels = []
    predicted_labels = []
    features = []
    for batch in eval_data:
      if self.adaptor is not None:
        logging.info('Running adaptation algorithm for evaluation.')
        property_label = batch[self.config.data.label_property]
        predicted_label, _ = self.adaptor.run(
            self.eval_batch, property_label, inputs=batch, rng=rng)
      else:
        predicted_label, _ = self.eval_batch(params, state, batch, rng)
      label = batch[self._label]
      feature = batch[self.config.data.label_property]

      # Concatenate along the pmapped direction.
      labels.append(jnp.concatenate(label))
      features.append(jnp.concatenate(feature))
      predicted_labels.append(jnp.concatenate(predicted_label))

    # And finally concatenate along the first dimension.
    labels = jnp.concatenate(labels)
    features = jnp.concatenate(features)
    predicted_labels = jnp.concatenate(predicted_labels)

    # Compute the metrics.
    results = {}
    for metric in self.config.evaluation.metrics:
      logging.info('Evaluating metric %s.', str(metric))
      metric_fn = getattr(metrics, metric, None)
      results[metric] = metric_fn(labels, features, predicted_labels, None)

    # Dump all the results by saving pickled results to disk.
    out_dir = checkpointing.get_checkpoint_dir(FLAGS.config)
    dataset = self.config.data.test_kwargs.load_kwargs.subset
    results_path = os.path.join(out_dir, f'results_{dataset}')
    if not tf.io.gfile.exists(results_path):
      tf.io.gfile.makedirs(results_path)

    # Save numpy arrays.
    with tf.io.gfile.GFile(
        os.path.join(results_path, 'results.pkl'), 'wb') as f:
      # Using protocol 4 as it's the default from Python 3.8 on.
      pickle.dump({'all_labels': labels, 'all_features': features,
                   'all_predictions': predicted_labels}, f, protocol=4)

    return results

  def _eval_batch(self, params: optax.Params, state: optax.OptState,
                  inputs: data_utils.Batch,
                  rng: chex.PRNGKey
                  ) -> Tuple[data_utils.ScalarDict, chex.Array]:
    """Evaluates a batch."""

    logits, _ = self.forward.apply(
        params, state, rng, inputs, is_training=False)

    inputs['one_hot_label'] = hk.one_hot(
        inputs[self._label], self.config.data.n_classes)
    (_, logits), _ = self.learner_fn.apply(params, state, rng, inputs)

    softmax_predictions = jax.nn.softmax(logits, axis=-1)
    predicted_label = jnp.argmax(softmax_predictions, axis=-1)

    return predicted_label, logits
