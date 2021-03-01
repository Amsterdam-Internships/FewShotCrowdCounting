# DEATH to the CNNs

The field of computer vision received a wild contribution from researches at Facebook, whom succeeded in training a fully transformer based architecture (DeiT) to do image classification on ImageNet. In this repository, I build upon their findings by adjusting their architecture to perform regression in the context of crowd counting. That is, I transform the embedding vectors from DeiT such that a density map can be constructed. Learning is then performed as usual. I can highly recommend [this](https://arxiv.org/abs/2012.12877) read.

Apart from a potential groundbreaking paper, the reptilian overlords of Facebook also blessed us peasants with several variants of their DeiT architecture, including the weights of the trained models and all code to evaluate and train the models. Hence, just like CSRNet and almost all architecures in crowd counting, we have a solid model that can be regarded as a 'general feature extractor'. Fine-tuning the weights of the model to do crowd counting can now be done in a day!

Preliminary results indicate that transformers have major potentials and that crowd counting competes with modern CNN methods. On some datasets, DeiT showed its superiority over CSRNet. On others it lost (just slightly) to CSRNet.

Current work focusses on simply making a solid code-base from which we can expand upon. Furthermore, since this is the first work to do regression in this way with transformers, I will perform extensive experiments to find what modifications work and which dont. Afterwards, I will use this architecture as a baseline in my persuit of a 'general' model that can quickly adapt to novel scenes it has never seen before. For this, I plan to implement and test two so called '[meta-learning](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)' algorithms: 1) [MAML](https://arxiv.org/abs/1703.03400) and 2) [Meta-SGD](https://arxiv.org/abs/1707.09835)

I write these meta-learning algorithms is a general, model-agnostic, way, such that it is almost trivial to swap out DeiT for any deep learning crowd counting architecture. This, because there is still a possibility that DeiT shows major defects on some part later in my thesis. We can then change the architecture without much trouble for the practical viewpoint of this project.

The end goal of this project is to have one or more of the DeiT architectures that can readily adjust to new scenes with just 1 to 5 annotated examples of this scene.  

## Research questions and research directions

<p align="center"> <i> RQ 1: How can transformers be utilised to generate density maps of crowds in images? </i> </p> 

I plan to perform an ablation study with DeiT on the Shanghaitech part B dataset. With this I expect to find how to properly train a transformer-based crowd counting model. I also plan to extend these findings to the datasets of the Municipality, although that will probably not be part of my thesis.  

<p align="center"> <i> RQ 2: Do transformer-based models generalise better than CNN-based models? </i> </p> 

Zero-shot adaptation, or transfer learning, in which we use a model pretrained on some dataset on another dataset without fine-tuning. This result will be especially usefull for scenarios where we dont have the time or resources to train a new model (as we have already found that fine-tuning with limited data is not sufficient). So far, I did find some settings where DeiT provides far superior transfer learning performance. Can we maximize transfer-learning performance? When does it work better and when does it fail?

Furthermore, the holy grail would be a model so good that no fine-tuning is necessary for adequate performance. We know no existing method is able to provide this. Can transformers be the key to supremacy? It's a far stretch, but nevertheless interesting to see how far we can push DeiT.

<p align="center"> <i> RQ 3: Do transformer-based models provide better few-shot scene adaptation performance than CNN-based models? </i> </p> 

Using Meta-SGD or MAML, can we train a model that adapts better to unseen scenarios than standard pre-trained models? [one work](https://arxiv.org/pdf/2002.00264.pdf) did show very promosing results, although I failed to reproduce their work so far. I start to lose my trust in these methods, though it's the direction of my research and so I want to conclude formally how well these methods perform. Should these experiments prove fruitful, we would no longer be required to annotate 100+ images for each new camera we place.

