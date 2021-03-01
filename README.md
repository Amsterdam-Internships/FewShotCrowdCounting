# DEATH to the CNNs

The field of computer vision received a wild contribution from researches at Facebook, whom succeeded in training a fully transformer based architecture (DeiT) to do image classification on ImageNet. In this repository, I build upon their findings by adjusting their architecture to perform regression in the context of crowd counting. That is, I transform the embedding vectors from DeiT such that a density map can be constructed. Learning is then performed as usual. I can highly recommend [this](https://arxiv.org/abs/2012.12877) read.

Apart from a potential groundbreaking paper, the reptilian overlords of Facebook also blessed us peasants with several variants of their DeiT architecture, including the weights of the trained models and all code to evaluate and train the models. Hence, just like CSRNet and almost all architecures in crowd counting, we have a solid model that can be regarded as a 'general feature extractor'. Fine-tuning the weights of the model to do crowd counting can now be done in a day!

Preliminary results indicate that transformers have major potentials and that crowd counting competes with modern CNN methods. On some datasets, DeiT showed its superiority over CSRNet. On others it lost (just slightly) to CSRNet.

Current work focusses on simply making a solid code-base from which we can expand upon. Furthermore, since this is the first work to do regression in this way with transformers, I will perform extensive experiments to find what modifications work and which dont. Afterwards, I will use this architecture as a baseline in my persuit of a 'general' model that can quickly adapt to novel scenes it has never seen before. For this, I plan to implement and test two so called '[meta-learning](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)' algorithms: 1) [MAML](https://arxiv.org/abs/1703.03400) and 2) [Meta-SGD](https://arxiv.org/abs/1707.09835)

I write these meta-learning algorithms is a general, model-agnostic, way, such that it is almost trivial to swap out DeiT for any deep learning crowd counting architecture. This, because there is still a possibility that DeiT shows major defects on some part later in my thesis. We can then change the architecture without much trouble for the practical viewpoint of this project.

The end goal of this project is to have one or more of the DeiT architectures that can readily adjust to new scenes with just 1 to 5 annotated examples of this scene.  

## Research questions and research directions

<p align="center"> <i> RQ 1: How can transformers be utilised to generate density maps of crowds in images? </i> </p> 

<p align="center"> <i> RQ 2: Do transformer-based models generalise better than CNN-based models? </i> </p> 

<p align="center"> <i> RQ 3: Do transformer-based models provide better few-shot scene adaptation performance than CNN-based models? </i> </p> 
