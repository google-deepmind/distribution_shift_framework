# Distribution Shift Framework

This repository contains the code of the distribution shift framework presented
in [A Fine-Grained Analysis on Distribution Shift](https://openreview.net/forum?id=Dl4LetuLdyK)
(Wiles et al., 2022).

## Contents

The framework allows to train models with different training methods on
datasets undergoing specific kinds of distribution shift.

### Training Methods

Currently the following training methods are supported (by setting the
`algorithm` [config option](#config-options)):

* Empirical Risk Minimization (**ERM**, [Vapnik, 1992](https://papers.nips.cc/paper/1991/hash/ff4d5fbbafdf976cfdc032e3bde78de5-Abstract.html))
* Invariant Risk Minimization (**IRM**, [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
* Deep Correlation Alignment (**Deep CORAL**, [Sun & Saenko, 2016](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35))
* Domain-Adversarial Training of Neural Networks (**DANN**, [Ganin et al., 2016](https://jmlr.org/papers/v17/15-239.html))
* Style-Agnostic Networks (**SagNet**, [Nam et al., 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Nam_Reducing_Domain_Gap_by_Reducing_Style_Bias_CVPR_2021_paper.html))
* (Batch Normalization Adaption (**BN-Adapt**, [Schneider et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html))
* Just Train Twice (**JTT**, [Liu et al., 2021](http://proceedings.mlr.press/v139/liu21f.html))
* Inter-domain Mixup (**MixUp**, [Gulrajani & LopezPaz, 2021](https://openreview.net/forum?id=lQdXeXDoWtI))

### Model Architectures

The `model` [config option](#config-options) can be set to one of the following
architectures

* ResNet18, ResNet50, ResNet101 ([He et al., 2016](https://ieeexplore.ieee.org/document/7780459))
* MLP ([Vapnik, 1992](https://papers.nips.cc/paper/1991/hash/ff4d5fbbafdf976cfdc032e3bde78de5-Abstract.html))

### Datasets

You can train on the following datasets (by setting the `dataset_name`
[config option.](#config-options)):

* dSprites ([Matthey et al., 2017](https://github.com/deepmind/dsprites-dataset))
* SmallNorb ([LeCun et al., 2004](https://ieeexplore.ieee.org/document/1315150))
* Shapes3D ([Burgess & Kim, 2018](https://github.com/deepmind/3d-shapes))

Each dataset has a task (e.g. shape prediction on dSprites, set with the `label`
[config option](#config-options)) and a set of properties (e.g. the colour of
the shape in dSprites, set with the `property_label`
[config option](#config-options)).

### Distribution Shift Scenarios

You can evaluate your model on different conditions by varying the distribution
of labels and properties in the configs. For each part of the distribution,
you then assign a probability of sampling from that part of the distribution.

* **Unseen data shift** (`ood`): Some parts of the distribution of the property
    are unseen at training time (e.g. certain colours may be unseen in
    dSprites).
* **Spurious correlation** (`correlated`): Some property is correlated with the
    label at training time but not at test (e.g. all circles are red in
    training).
* **Low data drift** (`lowdata`): Certain combinations of label and property are seen at a
    a lower rate during training while they are uniformly distributed during
    test.

Additionally you can modify these scenarios with two conditions:

* **Label noise** (`noise`): A certain percentage of the training labels are
    corrupted.
* **Fixed dataset size** (`fixeddata`): We reduce the total training dataset
    size to a fixed amount.

These scenarios can be set through the `test_case`
[config option.](#config-options)) with the keywords in parenthesis and an
optional modifier separated by a full stop, e.g. `lowdata.noise` for low data
drift with added label noise.

### Future Additions

We plan to add additional methods, models and datasets from the paper as well
as the raw results from all the experiments.

## Usage Instructions

### Installing

The following has been tested using Python 3.9.9.

For GPU support with JAX, edit `requirements.txt` before running `run.sh`
(e.g., use `jaxline==0.1.67+cuda111`). See JAX's installation
[instructions](https://github.com/google/jax#installation) for more details.

Execute `run.sh` to create and activate a virtualenv, install all necessary
dependencies and run a test program to ensure that you can import all the
modules.

```
# Run from the parent directory.
sh distribution_shift_framework/run.sh
```


### Running the Code

To train a model, use this virtualenv:

```
source /tmp/distribution_shift_framework/bin/activate
```

and then run

```
python3 -m distribution_shift_framework.classification.experiment \
--jaxline_mode=train \
--config=distribution_shift_framework/classification/config.py
```

For evaluation run

```
python3 -m distribution_shift_framework.classification.experiment \
--jaxline_mode=eval \
--config=distribution_shift_framework/classification/config.py
```

### Config Options {#config-options}

Common changes can be done through an options string following the config file.
The following options are available:

* `algorithm`: What training method to use for training.
* `model`:: The model architecture to evaluate.
* `dataset_name`: The name of the dataset.
* `test_case`: Which of the distribution shift scenarios to set up.
* `label`: The label we're predicting.
* `property_label`: Which property is treated as in or out of
  distribution (for the ood test_case), is correlated with the label
  (for the correlated setup) and is treated as having a low data region
  (for the low_data setup).
* `number_of_seeds`: How many seeds to sweep over.
* `batch_size`: Batch size used for training and evaluation.
* `training_steps`: How many steps to train for.
* `pretrained_checkpoint`: Path to a checkpoint for a pretrained model.
* `overwrite_image_size`: Height and width to resize the images to. 0 means
  no resizing.
* `eval_specific_ckpt`: Path to a checkpoint for a one time evaluation.
* `wids`: Which wids of the checkpoint to look at.
* `sweep_index`: Which experiment from the sweep to run.
* `use_fake_data`: Whether to use fake data for testing.


Multiple options need to be separated by commas. An example would be

```
python3 -m distribution_shift_framework.classification.experiment \
--jaxline_mode=train \
--config=distribution_shift_framework/classification/config.py:algorithm=SagNet,test_case=lowdata.noise,model=truncatedresnet18,property_label=label_object_hue,label=label_shape,dataset_name=shapes3d
```

Which would train a **truncated ResNet18** with the **SagNet** algorithm in the
**low data** setting with added **label noise** on the **Shapes3D** dataset.
**Shape** is used as the label for classification while **object hue** is used
as the property that the distribution shifts over.

### Sweeps

By default the program generates sweeps over multiple hyper-parameters depending
on the chosen training method, dataset and distribution shift scenario. The
`sweep_index` option lets you choose which of the configs in the sweep you want
to run.

## Citing this work

If you use this code (or any derived code) in your work, please cite the
accompanying paper:

```
@inproceedings{wiles2022fine,
  title={A Fine-Grained Analysis on Distribution Shift},
  author={Olivia Wiles and Sven Gowal and Florian Stimberg and Sylvestre-Alvise Rebuffi and Ira Ktena and Krishnamurthy Dj Dvijotham and Ali Taylan Cemgil},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=Dl4LetuLdyK}
}
```

## License and Disclaimer

Copyright 2022 DeepMind Technologies Limited.

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the License. You may obtain
a copy of the Apache 2.0 license at

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

All non-code materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY License). You may obtain a copy of the CC-BY
License at:

[https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode)

You may not use the non-code portions of this file except in compliance with the
CC-BY License.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

This is not an official Google product.
