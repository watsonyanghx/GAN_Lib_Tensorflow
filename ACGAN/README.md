# ACGAN

Code for paper: [Conditional Image Synthesis With Auxiliary Classifier GANs](https://openreview.net/forum?id=BkDDM04Ke)


I'm working on it, but I don't have available GPU now .... and by the way, the GPU cloud (AWS, Tencent and so on) is so so so so expensive!!!!!!



### How to run:

``` python
# cd to `ACGAN` folder and run command bellow
CUDA_VISIBLE_DEVICES=0 python ACGAN/train.py \
  --batch_size=64 \
  --mode='train' \
  --max_iter=100000 \
  --loss_type='WGAN-GP' \
  --n_dis=5 \
  --acgan_scale_G=0.1 \
  --data_dir='/home/yhx/sn_gan/cifar-10'

```



# LSUN

### TODO



