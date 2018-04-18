# SNGAN

Code for paper: [Spectral Normalization for Generative Adversarial Networks.](https://openreview.net/forum?id=B1QRgziT-)


**TODO**

I didn't notice the [Prjection Discriminator](https://openreview.net/forum?id=ByS1VpgRZ&noteId=r14W7yTrf), I plan to add this feature to this repo.


# cifar-10


### Results

The image generated:

![sample](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/SNGAN/img/samples_99999.png)

There are 10 columns with each corresponding to one label, in the same column, **only** the random noise is different.


The inception_50k curve:

![inception_50k](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/SNGAN/img/inception_50k.jpg)


The inception_50k_std curve:

![inception_50k_std](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/SNGAN/img/inception_50k_std.jpg)


The generator loss curve:

![g_cost](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/SNGAN/img/g_cost.jpg)


The discriminator loss curve:

![d_cost](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/SNGAN/img/d_cost.jpg)


### Inception Score

The **best inception_50k** is : `8.432400703430176 ± 0.1185135617852211`, the inception score may be improved by tuning hyperparameters or continuing training.


**Note:**

1. This is **Conditional** generation on cifar10, while in the paper, it is **Unconditional** generation and the inception score reported is: `8.22 ± .05`.

2. It's hard to say whether the improvement has something to do with the supervision from label, although more information is introdeced, it also means the Generator has to generate image consitent with the label, in other words, the generation process becomes harder. So, the **Unconditional** generation may or may not get better results, you can have a try and see what will happen.


### How to run:

Change [this line](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/SNGAN/gan_cifar_resnet.py#L72) to use 1 or 2 GPU. Hyperparameters is set according to what talked in paper.

Make sure the hyperparameters (from [this line](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/SNGAN/gan_cifar_resnet.py#L40)) are what you want before running the command bellow.

``` python
# cd to `SNGAN` folder and run command bellow
CUDA_VISIBLE_DEVICES=0,1 python gan_cifar_resnet.py

```


### Pretrained model

The pretrained model can be downloaded [here](https://www.dropbox.com/sh/ce5nlk0v0tgq0ah/AABEvy3T2X1WFkYqCV5ze59ga?dl=0).

You can use the pretrained model to generate images, or you can use this as an initialization to continue training (training from start can be slow, about 2 days on single 1080Ti).


### Other issue

Since this is **Conditional** generation, the label information is introdeced by [Conditional Batch Normalization](https://openreview.net/forum?id=BJO-BuT1g) in `Generator` and concatenating label information with intermediate feature map in `Discriminator` (See [code](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/SNGAN/gan_cifar_resnet.py#L283) in `Discriminator` for clarification).

I have tried concatenating label information with noise `z` in `Generator`, but the result is not good, see bellow for more details:


- `Concatenate label information in G + Conditional Batch Norm in G` : use small learning rate (e.g. 0.0001) can convergece, but become slow.

- `Concatenate label information in G + Batch Norm in G` : replace [Conditional Batch Normalization](https://openreview.net/forum?id=BJO-BuT1g) with [Batch Normalization](http://proceedings.mlr.press/v37/ioffe15.pdf), the training is similar with above.


I didn't tune hyperparameters, so the results may get better than I tried, with hyperparameter set properly.

You can play with the network architecture of Generator/Discriminator besides tuning hyperparameters, and see what will happen. 

Good luck and have fun!


# ImageNet

### TODO



