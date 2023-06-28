# MorphGANFormer

This is the offcial implementation of paper 'MorphGANFormer: Transformer-based Face Morphing and De-Morphing'.
Code will be coming soon.

![arch](fig/Morph_latentvs2.png)

### Link: 
[[PDF]](https://arxiv.org/pdf/2302.09404.pdf)
[[Arxiv]](https://arxiv.org/abs/2302.09404)


This repository contains the implementation of:
* MorphGANFormer 
* Morphing and De-morphing

Our model is based on the paper:  GANformer: Generative Adversarial Transformers 
[[Github]](https://github.com/dorarad/gansformer)
[[Arxiv]](https://arxiv.org/abs/2103.01209)


# Introduction
* Inspired by GANformer, we introduce a transformer-based face morphing algorithm. 
* Special loss functions are designed to support the optimization of face morphing process. 
* We extend the study of transformer-based face morphing to demorphing by presenting an effective defense strategy with access to a reference image using the same generator of MorphGANFormer. Such demorphing is conceptually similar to unmixing of hyperspectral images but operates in the latent (instead of pixel) space. 

# Environment
- Python 3.6 or 3.7 are supported.
- Pytorch >= 1.8.
- CUDA 10.0 toolkit and cuDNN 7.5.
- See [`requirements.txt`](requirements.txt) for the required python packages and run `pip install -r requirements.txt` to install them.

# Face Morphing Pipeline
![arch](fig/Morph_pipeline.png)

# Face De-Morphing Pipeline
![arch](fig/Morph_demorph_pipeline.png)

# Face Latent code Optimization
![arch](fig/Morph_latentcode.png)
