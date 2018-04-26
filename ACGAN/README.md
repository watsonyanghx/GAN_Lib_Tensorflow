# ACGAN

Code for paper: [Conditional Image Synthesis With Auxiliary Classifier GANs](https://openreview.net/forum?id=BkDDM04Ke)


# cifar-10

### Results

The image generated:

![sample](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/ACGAN/img/samples.png)

There are 10 columns with each corresponding to one label, in the same column, **only** the random noise is different.


The inception_50k curve:

![inception_50k](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/ACGAN/img/inception_50k.jpg)


The inception_50k_std curve:

![inception_50k_std](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/ACGAN/img/inception_50k_std.jpg)


### Inception Score

The **inception_50k** is : `7.865674018859863, ± 0.09258274734020233`, the inception score can be improved by changint nerwork architecture or tuning hyperparameters.

I really don't konw how to improve it, if you know how to fix this, a PR is welcomed.


### How to run:

``` python
# cd to `ACGAN` folder and run command bellow
CUDA_VISIBLE_DEVICES=0 python ACGAN/train.py \
  --batch_size=64 \
  --mode='train' \
  --max_iter=100000 \
  --loss_type='WGAN-GP' \
  --n_dis=5 \
  --acgan_scale_G=0.4 \
  --data_dir='/home/yhx/gan_lib/cifar-10'

// adam超参数改为使用wgan-gp的设定
```


### Pretrained model

I didn't upload the pre-trained model, because this model can be improved by changing the network architecture or tuning the hyptermeters. 

If you do need the pre-trained model, please open an issue, and I will upload it.


# LSUN

### TODO
