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

"""Config for imagenet experiment."""

import functools
from typing import Any, Callable, List, Mapping, Optional, Tuple

import chex
from distribution_shift_framework.configs import disentanglement_config
from distribution_shift_framework.core import adapt
from distribution_shift_framework.core import adapt_train
from distribution_shift_framework.core import algorithms
from distribution_shift_framework.core import hyper
from distribution_shift_framework.core.datasets import data_utils
from distribution_shift_framework.core.model_zoo import resnet
from distribution_shift_framework.core.pix import postprocessing
import haiku as hk
from jaxline import base_config
import ml_collections



DATASETS = ('dsprites', 'small_norb', 'shapes3d')
LEARNERS = algorithms.__all__
ADAPTERS = ('BNAdapt',)
TRAIN_ADAPTERS = ('JTT',)
POSTPROCESSORS = ('mixup',)
ALGORITHMS = LEARNERS + ADAPTERS + TRAIN_ADAPTERS + POSTPROCESSORS


ConfigAndSweeps = Tuple[ml_collections.ConfigDict, List[hyper.Sweep]]

_EXP = 'config.experiment_kwargs.config'


def parse_options(
    options: str,
    defaults: Mapping[str, Any],
    types: Optional[Mapping[str, Callable[[str], Any]]] = None
) -> Mapping[str, Any]:
  """Parse a "k1=v1,k2=v2" option string."""
  if not options:
    return defaults
  if types is None:
    types = {}
  else:
    types = dict(**types)
  for k, v in defaults.items():
    if k not in types:
      types[k] = type(v)
  kwargs = dict(t.split('=', 1) for t in options.split(','))
  for k, v in kwargs.items():
    if k in types:  # Default type is `str`.
      kwargs[k] = ((v in ('True', 'true', 'yes')) if types[k] == bool
                   else types[k](v))
  # Only allow options where defaults are specified to avoid typos.
  for k in kwargs:
    if k not in defaults:
      raise ValueError('Unknown option `%s`.' % k)
  for k, v in defaults.items():
    if k not in kwargs:
      kwargs[k] = v
  return kwargs


def get_config(options: str = '') -> ml_collections.ConfigDict:
  """Return config object for training.

  Args:
    options: A list of options that are comma separated with:
      key1=value1,key2=value2. The actual key value pairs are the following:

      dataset_name -- The name of the dataset.
      model -- The model to evaluate.
      test_case -- Which of ood or correlated setups to run.
      label -- The label we're predicting.
      property_label -- Which property is treated as in or out of
        distribution (for the ood test_case), is correlated with the label
        (for the correlated setup) and is treated as having a low data region
        (for the low_data setup).
      algorithm -- What algorithm to use for training.
      number_of_seeds -- How many seeds to evaluate the models with.
      batch_size -- Batch size used for training and evaluation.
      training_steps -- How many steps to train for.
      pretrained_checkpoint -- Path to a checkpoint for a pretrained model.
      overwrite_image_size -- Height and width to resize the images to. 0 means
        no resizing.
      eval_specific_ckpt -- Path to a checkpoint for a one time evaluation.
      wids -- Which wids of the checkpoint to look at.
      sweep_index -- Which experiment from the sweep to run.
      use_fake_data -- Whether to use fake data for testing.
  Returns:
    ConfigDict: A dictionary of parameters.
  """
  options = parse_options(
      options,
      defaults={
          'dataset_name': 'dsprites',
          'model': 'resnet18',
          'test_case': 'ood',
          'label': 'label_shape',
          'property_label': 'label_color',
          'algorithm': 'ERM',
          'number_of_seeds': 1,
          'batch_size': 128,
          'training_steps': 100_000,
          'pretrained_checkpoint': '',
          'overwrite_image_size': 0,  # Zero means no resizing.
          'eval_specific_ckpt': '',
          'wids': '1-1',
          'sweep_index': 0,
          'use_fake_data': False,
      })
  assert options['dataset_name'] in DATASETS
  assert options['algorithm'] in ALGORITHMS
  if options['algorithm'] in LEARNERS:
    learner = options['algorithm']
    adapter = ''
    train_adapter = ''
    postprocessor = ''
  else:
    learner = 'ERM'
    if options['algorithm'] in ADAPTERS:
      adapter = options['algorithm']
    elif options['algorithm'] in TRAIN_ADAPTERS:
      train_adapter = options['algorithm']
    elif options['algorithm'] in POSTPROCESSORS:
      postprocessor = options['algorithm']
  config = base_config.get_base_config()
  config.random_seed = 0
  config.checkpoint_dir = '/tmp'
  config.train_checkpoint_all_hosts = False

  training_steps = options['training_steps']

  config.experiment_kwargs = ml_collections.ConfigDict()

  exp = config.experiment_kwargs.config = ml_collections.ConfigDict()
  exp.use_fake_data = options['use_fake_data']
  exp.enable_double_transpose = False

  # Training.
  exp.training = ml_collections.ConfigDict()
  exp.training.use_gt_images = False
  exp.training.save_images = False
  exp.training.batch_size = options['batch_size']
  exp.training.adversarial_weight = 1.
  exp.training.label_noise = 0.0

  # Evaluation.
  exp.evaluation = ml_collections.ConfigDict()
  exp.evaluation.batch_size = options['batch_size']
  exp.evaluation.metrics = ['top1_accuracy']

  # Optimizer.
  exp.optimizer = ml_collections.ConfigDict()
  exp.optimizer.name = 'adam'
  exp.optimizer.kwargs = dict(learning_rate=0.001)

  # Data.
  exp.data = ml_collections.ConfigDict()
  if data_utils.is_disentanglement_dataset(options['dataset_name']):
    exp.data = disentanglement_config.get_renderers(
        options['test_case'], dataset_name=options['dataset_name'],
        label=options['label'],
        property_label=options['property_label'])
    data_sweep = disentanglement_config.get_renderer_sweep(
        options['test_case'])
  else:
    dataset_name = options['dataset_name']
    raise ValueError(f'Unsupported dataset {dataset_name}')

  if exp.use_fake_data:
    # Data loaders skip valid and test samples and default values are so high
    # that we would need to generate too many fake datapoints.
    batch_size = options['batch_size']
    if options['dataset_name'] in ('dsprites', 'shapes3d'):
      exp.data.train_kwargs.load_kwargs.dataset_kwargs.valid_size = batch_size
      exp.data.train_kwargs.load_kwargs.dataset_kwargs.test_size = batch_size
      exp.data.test_kwargs.load_kwargs.valid_size = batch_size
      exp.data.test_kwargs.load_kwargs.test_size = batch_size
    elif options['dataset_name'] == 'small_norb':
      exp.data.train_kwargs.load_kwargs.dataset_kwargs.valid_size = batch_size
      exp.data.test_kwargs.load_kwargs.valid_size = batch_size

  # Model.
  model = options['model']
  exp.model, model_sweep = globals()[f'get_{model}_config'](
      num_classes=exp.data.n_classes, resize_to=options['overwrite_image_size'])
  exp.pretrained_checkpoint = options['pretrained_checkpoint']

  # Learning algorithm.
  exp.training.algorithm, learner_sweep = get_learner(
      learner, model, exp.data.n_classes)

  # Test time adaptation.
  if adapter:
    exp.adapter = get_adapter(adapter, exp.data.n_properties)
  else:
    exp.adapter = ml_collections.ConfigDict()

  # Adapt training parameters and state.
  if train_adapter:
    exp.training.learn_adapt = get_train_adapter(
        train_adapter, training_steps=training_steps)
  else:
    exp.training.learn_adapt = ml_collections.ConfigDict()

  # Postprocessing.
  if postprocessor:
    exp.postprocess = get_postprocessing_step(postprocessor)
  else:
    exp.postprocess = ml_collections.ConfigDict()

  if exp.data.train_kwargs.load_kwargs.get('shuffle_pre_sampling', False):
    exp_train_kwargs = 'config.experiment_kwargs.config.data.train_kwargs.'
    seeds = list(range(options['number_of_seeds']))
    random_seedsweep = hyper.zipit([
        hyper.sweep('config.random_seed', seeds),
        hyper.sweep(f'{exp_train_kwargs}load_kwargs.shuffle_pre_sample_seed',
                    seeds)])
  else:
    random_seedsweep = hyper.sweep('config.random_seed',
                                   list(range(options['number_of_seeds'])))

  all_sweeps = hyper.product(
      [random_seedsweep] + [data_sweep] + model_sweep + learner_sweep)

  dataset_name = options['dataset_name']

  config.autoxprof_warmup_steps = 5
  config.autoxprof_measure_time_seconds = 50

  # Use so get consistency between different models with different speeds.
  config.interval_type = 'steps'

  config.training_steps = training_steps
  config.log_train_data_interval = 1_000
  config.log_tensors_interval = 1_000
  config.save_checkpoint_interval = 1_000
  config.eval_specific_checkpoint_dir = options['eval_specific_ckpt']
  if options['eval_specific_ckpt']:
    min_wid, max_wid = [int(w) for w in options['wids'].split('-')]
    config.eval_only = True
    config.one_off_evaluate = True
    all_sweeps = hyper.product([hyper.zipit([
        hyper.sweep('config.eval_specific_checkpoint_dir',
                    [options['eval_specific_ckpt'].format(wid=w)
                     for w in range(min_wid, max_wid+1)]),
        all_sweeps])])

  else:
    config.eval_only = False
  config.best_model_eval_metric = 'top1_accuracy'

  config.update_from_flattened_dict(all_sweeps[options['sweep_index']],
                                    'config.')

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config


def get_postprocessing_step(postprocessing_name: str
                            ) -> ml_collections.ConfigDict:
  """Config for postprocessing steps."""
  postprocess = ml_collections.ConfigDict()
  postprocess.fn = getattr(postprocessing, postprocessing_name)
  postprocess.kwargs = ml_collections.ConfigDict()

  if postprocessing_name == 'mixup':
    postprocess.kwargs.alpha = 0.2
    postprocess.kwargs.beta = 0.2

  return postprocess


def get_train_adapter(adapter_name: str, training_steps: int
                      ) -> ml_collections.ConfigDict:
  """Config for adapting the training parameters."""
  adapter = ml_collections.ConfigDict()
  adapter.fn = getattr(adapt_train, adapter_name)
  adapter.kwargs = ml_collections.ConfigDict()

  if adapter_name == 'JTT':
    adapter.kwargs.lmbda = 20
    adapter.kwargs.num_steps_in_first_iter = training_steps // 2
  return adapter


def get_adapter(adapt_name: str, num_properties: int
                ) -> ml_collections.ConfigDict:
  """Config for how to adapt the model at test time."""
  adapter = ml_collections.ConfigDict()
  adapter.fn = getattr(adapt, adapt_name)
  adapter.kwargs = ml_collections.ConfigDict(dict(n_properties=num_properties))
  adapter.num_adaptation_steps = 1_000
  return adapter


def get_learner(learner_name: str,
                model_name: str,
                num_classes: int = 10) -> ConfigAndSweeps:
  """Config for which learning algorithm to use."""
  learner = ml_collections.ConfigDict()
  learner.fn = getattr(algorithms, learner_name)
  learner.kwargs = ml_collections.ConfigDict()

  learner_sweep = []
  exp_algthm = f'{_EXP}.training.algorithm'
  if learner_name == 'IRM':
    learner.kwargs.lambda_penalty = 1.
    learner_sweep = [
        hyper.sweep(f'{exp_algthm}.kwargs.lambda_penalty',
                    [0.01, 0.1, 1, 10])
    ]
  elif learner_name == 'DANN':
    learner.kwargs.mlp_output_sizes = ()
    exp = f'{_EXP}.training'
    learner_sweep = [
        hyper.sweep(f'{exp}.adversarial_weight',
                    [0.01, 0.1, 1, 10]),
        hyper.sweep(f'{exp_algthm}.kwargs.mlp_output_sizes',
                    [(64, 64)])
    ]
  elif learner_name == 'CORAL':
    learner.kwargs.coral_weight = 1.
    learner_sweep = [
        hyper.sweep(f'{exp_algthm}.kwargs.coral_weight',
                    [0.01, 0.1, 1, 10])
    ]
  elif learner_name == 'SagNet':
    if model_name == 'truncatedresnet18':
      learner.kwargs.content_net_kwargs = ml_collections.ConfigDict(dict(
          output_sizes=(num_classes,)))
      learner.kwargs.style_net_kwargs = ml_collections.ConfigDict(dict(
          output_sizes=(num_classes,)))
    else:
      learner.kwargs.content_net_kwargs = ml_collections.ConfigDict(dict(
          output_sizes=(64, 64, num_classes)))
      learner.kwargs.style_net_kwargs = ml_collections.ConfigDict(dict(
          output_sizes=(64, 64, num_classes)))
  return learner, learner_sweep


def _get_resizer(size: Optional[int]) -> Callable[[chex.Array], chex.Array]:
  if size is not None and size > 0:
    return functools.partial(data_utils.resize, size=(size, size))
  return lambda x: x


def get_mlp_config(n_layers: int = 4, n_hidden: int = 256,
                   num_classes: int = 10, resize_to: Optional[int] = None
                   ) -> ConfigAndSweeps:
  """Returns an MLP config and sweeps."""
  resize = _get_resizer(resize_to)
  mlp = ml_collections.ConfigDict(dict(
      constructor=hk.nets.MLP,
      kwargs=dict(output_sizes=[n_hidden] * n_layers + [num_classes]),
      preprocess=lambda x: resize(x).reshape((x.shape[0], -1))))
  sweep = hyper.sweep(f'{_EXP}.optimizer.kwargs.learning_rate',
                      [0.01, 0.001, 1e-4])
  return mlp, [sweep]


def get_resnet18_config(num_classes: int = 10,
                        resize_to: Optional[int] = None) -> ConfigAndSweeps:
  cnn = ml_collections.ConfigDict(dict(
      constructor=hk.nets.ResNet18,
      kwargs=dict(num_classes=num_classes),
      preprocess=_get_resizer(resize_to)))
  sweep = hyper.sweep(f'{_EXP}.optimizer.kwargs.learning_rate',
                      [0.01, 0.001, 1e-4])
  return cnn, [sweep]


def get_resnet50_config(num_classes: int = 10,
                        resize_to: Optional[int] = None) -> ConfigAndSweeps:
  cnn = ml_collections.ConfigDict(dict(
      constructor=hk.nets.ResNet50,
      kwargs=dict(num_classes=num_classes),
      preprocess=_get_resizer(resize_to)))
  sweep = hyper.sweep(f'{_EXP}.optimizer.kwargs.learning_rate',
                      [0.01, 0.001, 1e-4])
  return cnn, [sweep]


def get_resnet101_config(num_classes: int = 10,
                         resize_to: Optional[int] = None) -> ConfigAndSweeps:
  cnn = ml_collections.ConfigDict(dict(
      constructor=hk.nets.ResNet101,
      kwargs=dict(num_classes=num_classes),
      preprocess=_get_resizer(resize_to)))
  sweep = hyper.sweep(f'{_EXP}.optimizer.kwargs.learning_rate',
                      [0.01, 0.001, 1e-4])
  return cnn, [sweep]


def get_truncatedresnet18_config(
    num_classes: int = 10, resize_to: Optional[int] = None) -> ConfigAndSweeps:
  """Config for a truncated ResNet."""
  cnn = ml_collections.ConfigDict(dict(
      constructor=resnet.ResNet18,
      kwargs=dict(num_classes=num_classes),
      preprocess=_get_resizer(resize_to)))
  sweep = hyper.sweep(f'{_EXP}.optimizer.kwargs.learning_rate',
                      [0.01, 0.001, 1e-4])
  return cnn, [sweep]
