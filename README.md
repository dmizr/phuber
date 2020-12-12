# Can gradient clipping mitigate label noise?

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380//)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


A [PyTorch](https://pytorch.org/) implementation of the ICLR 2020 paper "[Can gradient clipping mitigate label noise?](https://openreview.net/pdf?id=rklB76EKPr)" by Menon et al.

This paper studies the robustness of gradient clipping to symmetric label noise, and proposes partially Huberised (PHuber) versions of standard losses, which perform well in the presence of label noise.

For the experiments, the following losses are also implemented:
- [Unhinged loss](https://arxiv.org/abs/1505.07634v1) (van Rooyen et al., NeurIPS 2015)
- [Generalized Cross Entropy loss](https://arxiv.org/abs/1805.07836v4) (Zhang & Sabuncu, NeurIPS 2018)

## Table of Contents
- [Dependencies](#dependencies)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Pretrained models](#pretrained)
- [References](#references)


## Dependencies
This project requires Python >= 3.8. Dependencies can be installed with:
```
pip install -r requirements.txt
```

## Training

This project uses [Hydra](https://hydra.cc/) to configure experiments. Configurations can be overriden through config files (in the `conf/` folder) and the command line. For more information, check out the [Hydra documentation](https://hydra.cc/docs/intro/).

With Hydra, configurations can be fully customized directly though the command line. To find out more about the configuration options, run:
```
python3 train.py --help
```

To run the experiments from the paper (72 different configurations), only 5 options need to be overriden.
These are:
- the dataset: `mnist, cifar10, cifar100` (e.g. `dataset=cifar100`)
- the model: `lenet, resnet50` (e.g. `model=resnet50`)
- the loss: `ce, gce, linear, phuber_ce, phuber_gce` (e.g. `loss=phuber_ce`)
- the label corruption probability ρ of the training set (e.g. `dataset.train.corrupt_prob=0.2`)
- the gradient clipping max norm (not used by default) (e.g. `hparams.grad_clip_max_norm=0.1`)

**Note:** When choosing a dataset and model, the hyper-parameters (e.g. number of epochs, batch size, optimizer, learning rate scheduler, ...) are automatically modified to use the configuration described by the authors in their experiments. If needed, these hyper-parameters can also be overriden through command line arguments.

#### Examples

Training LeNet on MNIST using cross-entropy loss and no label corruption:
```
python3 train.py dataset=mnist model=lenet loss=ce dataset.train.corrupt_prob=0.0
```

Training a ResNet-50 on CIFAR-10 using the partially Huberised cross-entropy loss (PHuber-CE) with τ=2, and label corruption probability ρ of 0.2:

```
python3 train.py dataset=cifar10 model=resnet50 loss=phuber_ce loss.tau=2 dataset.train.corrupt_prob=0.2
```

Training a ResNet-50 on CIFAR-100 using the Generalized Cross Entropy loss (GCE), label corruption probability ρ of 0.6, and with [Mixed Precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/):

```
python3 train.py dataset=cifar100 model=resnet50 loss=gce dataset.train.corrupt_prob=0.6 mixed_precision=true
```

 Training LeNet on MNIST using cross-entropy loss, and varying label corruption probability ρ (0.0, 0.2, 0.4 and 0.6). This uses [Hydra's multi-run flag](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run) for parameter sweeps:

```
python3 train.py --multirun dataset=mnist model=lenet loss=ce dataset.train.corrupt_prob=0.0,0.2,0.4,0.6
```

#### Run metrics and saved models
By default, run metrics are logged to [TensorBoard](https://www.tensorflow.org/tensorboard). In addition, the saved models, training parameters and training log can be found in the run's directory, in the `outputs/` folder.

## Evaluation
