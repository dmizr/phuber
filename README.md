# Can gradient clipping mitigate label noise?

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380//)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


A [PyTorch](https://pytorch.org/) implementation of the ICLR 2020 paper "[Can gradient clipping mitigate label noise?](https://openreview.net/pdf?id=rklB76EKPr)" by Menon et al.

This paper studies the robustness of gradient clipping to symmetric label noise, and proposes partially Huberised (PHuber) versions of standard losses, which perform well in the presence of label noise.

For the experiments, the following loss functions are also implemented:
- [Unhinged loss](https://arxiv.org/abs/1505.07634v1) (van Rooyen et al., NeurIPS 2015)
- [Generalized Cross Entropy loss](https://arxiv.org/abs/1805.07836v4) (Zhang & Sabuncu, NeurIPS 2018)
