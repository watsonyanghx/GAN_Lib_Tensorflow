#

Code for paper: [Spectral Normalization for Generative Adversarial Networks.](https://openreview.net/forum?id=B1QRgziT-)


## cifar-10


### How to run:

Change this [line]() to use 1/2 GPU. Hyperparameters is set according to paper.

``` python

CUDA_VISIBLE_DEVICES=0,1 python gan_cifar_resnet.py

```


### Training details

The image generated:

![]()

There are 10 columns with each corresponding to one label, in same column, only the random noise is different.


The generator loss curve:

![]()


The discriminator loss curve:

![]()


The inception_50k curve:

![]()


The inception_50k_std curve:

![]()


The best inception_50k is : `8.432400703430176 ± 0.1185135617852211`, the inception score may be improved by tuning hyperparameters or continuing training.

Note this is **Conditional** generation on cifar10, while in the paper, it is **Unconditional** generation and the inception score reported is: `8.22 ± .05`.

*Note:*
It's hard to say whether the improvement has something to do with the supervision from label, although more information is introdeced, it also means the Generator has to generate image consitent with the label, in other words, the generation process become harder. So, the **Unconditional** generation may or may not get better results, you can have a try and see what will happer.


### Pretrained model

[pretrained model](https://www.dropbox.com/sh/ce5nlk0v0tgq0ah/AABEvy3T2X1WFkYqCV5ze59ga?dl=0) can be downloaded.

You can use the pretrained model to generate images, or you can use this as initialization to continue training or training from start.


### Other issue

Since this is **Conditional** generation, the label information is introdeced by [Conditional Batch Normalization]() in Generator and concatenating label information with intermediate feature map in Discriminator (See [code]() in Discriminator for clarification).

I have tried concatenating label information with noise `z` in Generator, but the result is not good, see bellow for more details:


- `Concatenate label information in G + Conditional Batch Norm in G` : use small learning rate (e.g. 0.0001) can convergece, but become slow.

- `Concatenate label information in G + Batch Norm in G` : replace [Conditional Batch Normalization]() with [Batch Normalization](), the training is similar with above.


I didn't tune hyperparameters, so the results may get better than I tried, with hyperparameter set properly.


You can play with the network architecture of Generator/Discriminator besides the hyperparameters, and see what will happen. 

Good luck and have fun!


## ImageNet





