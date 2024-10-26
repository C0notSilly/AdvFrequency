# [NeurIPS'24] Towards Combating Frequency Simplicity-biased Learning for Domain Generalization

<!-- [![paper](https://img.shields.io/badge/arXiv-Paper-42FF33)](https://arxiv.org/abs/2403.02782) 
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://bimsarapathiraja.github.io/mccl-project-page/)   -->

This repository gives the official implementation of [Towards Combating Frequency Simplicity-biased Learning for Domain Generalization](https://neurips.cc/virtual/2024/poster/94907) (NeurIPS 2024)

## Introduction
Domain generalization methods aim to learn transferable knowledge from source domains that can generalize well to unseen target domains. Recent studies show that neural networks frequently suffer from a simplicity-biased learning behavior which leads to over-reliance on specific frequency sets, namely as frequency shortcuts, instead of semantic information, resulting in poor generalization performance. Despite previous data augmentation techniques successfully enhancing generalization performances, they intend to apply more frequency shortcuts, thereby causing hallucinations of generalization improvement. In this paper, we aim to prevent such learning behavior of applying frequency shortcuts from a data-driven perspective. Given the theoretical justification of models' biased learning behavior on different spatial frequency components, which is based on the dataset frequency properties, we argue that the learning behavior on various frequency components could be manipulated by changing the dataset statistical structure in the Fourier domain. Intuitively, as frequency shortcuts are hidden in the dominant and highly dependent frequencies of dataset structure, dynamically perturbating the over-reliance frequency components could prevent the application of frequency shortcuts. To this end, we propose two effective data augmentation modules designed to collaboratively and adaptively adjust the frequency characteristic of the dataset, aiming to dynamically influence the learning behavior of the model and ultimately serving as a strategy to mitigate shortcut learning.

<p align="center"><img src="af_teaser.jpg" width="500"></p>

## Setup the environment
Install PyTorch:
```bash
pip3 install torch torchvision torchaudio
```
Other versions of PyTorch could be install [here](https://pytorch.org/get-started/locally/).

## Train
After setting the training data and pretrained models accordingly, run the following line:
```bash
python train.py
```

## Shortcut Evaluation
The code of frequency shortcut evaluation could be found [here](https://github.com/nis-research/nn-frequency-shortcuts).

## Citation

```
@article{he2024towards,
  title={Towards Combating Frequency Simplicity-biased Learning for Domain Generalization},
  author={He, Xilin and Hu, Jingyu and Lin, Qinliang and Luo, Cheng and Xie, Weicheng and Song, Siyang and Khan, Muhammad Haris and Shen, Linlin},
  journal={arXiv preprint arXiv:2410.16146},
  year={2024}
}
```
