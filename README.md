# Vision Crowd Counting Transformers (ViCCT) for Cross-Scene Crowd Counting

![Example of density map regression](./misc/example_images/ViCCT_architecture.jpg?raw=true "ViCCT architecture")

This repository contains the code from the thesis project about crowd counting with fully transformer-based architectures. The following functions are provided:


* Standard training of ViCCT and CSRNet
* Meta learning with Meta-SGD for:
  *  CSRNet
  *  ViCCT
  *  SineNet (the toy example discussed in the [MAML](https://arxiv.org/abs/1703.03400) paper and the [Meta-SGD](https://arxiv.org/abs/1707.09835) paper)




## What is Crowd Counting?
As the name suggest, crowd counting involves estimating the number of people in a location. Most modern computer vision methods achieve this with density map regression. That is, given an image of a scene (for example, surveillance footage), predict a density map where each pixel indicated the accumalative density of all people at that point. The ground-truth density is often a Gaussian distribution around each person's head. If multiple persons are close to each other, the distributions overlap and the values are summed. We obtain the total count in a scene by integrating over the whole density map. The following shows an example of a scene, its ground-truth density map, and a model's prediction of this map, given only the image. Note that some of the density mass is outside the image frame due to the Gaussian distribution close to the edge of the image.

![Example of density map regression](./misc/example_images/density_example.jpg?raw=true "Example of density map regression")

## What is few-shot learning in the context of scene adaptation with crowd counting?
Few-shot learning means that the model must learn something with only a few training examples. Scene adaptation in crowd counting is that we have a model trained on one or more scens (e.g. one or more surveillance cameras), and that we wish to adjust the model to do crowd counting in a novel scene. Combine the two and you get that a model must adjust to a novel scene with just a few training examples. This is a non-trivial task due to the change in perspective, changes in lighting conditions, changes in people's appearance, changes in background, etc.


## Why do we need few-shot learning for scene adaptation?
The standard approach to obtain a model for a novel scene is to manually annotate many images of this scene, usually in the hundreds of images. This is extremely tedious and labour intensive. Should we succeed in obtaining a model that can adapt to new scenes with just a few images, we greatly reduce the required annotation time whenever we place a new camera.


# Using this repository:
First of all, the environment used with this project is provided in [`environment.yml`](./environment.yml). One can install this environment with 'conda env create -f environment.yml'.


To train any model, specify the parameters for the run in [`config.py`](./config.py), such as the model to train and the dataset to use. Note that the name for the dataset must match the folder name in [`datasets/standard`](./datasets/standard) for standard training and [`datasets/meta`](./datasets/meta) for meta learning. Dataset specific parameters are set in 'settings.py' in the folder of the corresponding dataset.

Training a model can be done with [`main_standard.py`](./main_standard.py) for standard training and [`main_meta.py`](./main_meta.py) for meta-learning.


# Acknowledgements

The code in this repository is heavily inspired by, and uses parts of, the Crowd Counting Code Framework ([`C^3-Framework`](https://github.com/gjy3035/C-3-Framework)). I also use and extend the code from the DeiT [`repository`](https://github.com/facebookresearch/deit) repository for the ViCCT models.

Code from [`this`](https://github.com/infinitemugen/MAML-Pytorch) repository about MAML in PyTorch is used for 1) our Meta-SGD implementation and 2) the SineNet implementation.



Important papers for this repository:
 - C^3-Framework [`paper`](https://arxiv.org/abs/1907.02724)
 - DeiT [`paper`](https://arxiv.org/abs/2012.12877)
 - CSRNet [`paper`](https://arxiv.org/abs/1802.10062)
 - Meta-SGD [`paper`](https://arxiv.org/abs/1707.09835)

