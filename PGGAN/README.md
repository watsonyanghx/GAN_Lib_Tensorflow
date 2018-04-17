#

Code for paper: [PGGAN](https://openreview.net/forum?id=B13njo1R-)

I'm working on it, but I don't have available GPU now .... and by the way, the GPU cloud (AWS, Tencent and so on) is so so so so expensive!!!!!!


## cifar-10


### How to run:

``` python

# resnet, change 'resnet' to 'nvidia' to use the network architecture proposed in paper.
# 4 × 4
CUDA_VISIBLE_DEVICES=1 python PGGAN/train.py \
  --batch_size=64 \
  --mode='train' \
  --image_size=4 \
  --data_dir=/home/yhx/gan_lib/cifar-10 \
  --model='resnet'


# 8 × 8
CUDA_VISIBLE_DEVICES=1 python PGGAN/train.py \
  --batch_size=64 \
  --mode='train' \
  --image_size=8 \
  --data_dir=/home/yhx/gan_lib/cifar-10 \
  --restore \
  --model='resnet' \
  --trans \
  --block_count=1


CUDA_VISIBLE_DEVICES=1 python PGGAN/train.py \
  --batch_size=64 \
  --mode='train' \
  --image_size=8 \
  --data_dir=/home/yhx/gan_lib/cifar-10 \
  --restore \
  --model='resnet' \
  --block_count=1


# 16 × 16  ################ on running ################
CUDA_VISIBLE_DEVICES=1 python PGGAN/train.py \
  --batch_size=64 \
  --mode='train' \
  --image_size=16 \
  --data_dir=/home/yhx/gan_lib/cifar-10 \
  --restore \
  --model='resnet' \
  --trans \
  --block_count=2

CUDA_VISIBLE_DEVICES=1 python PGGAN/train.py \
  --batch_size=64 \
  --mode='train' \
  --image_size=16 \
  --data_dir=/home/yhx/gan_lib/cifar-10 \
  --restore \
  --model='resnet' \
  --block_count=2


# 32 × 32
CUDA_VISIBLE_DEVICES=1 python PGGAN/train.py \
  --batch_size=64 \
  --mode='train' \
  --image_size=32 \
  --data_dir=/home/yhx/gan_lib/cifar-10 \
  --restore \
  --model='resnet' \
  --trans \
  --block_count=3

CUDA_VISIBLE_DEVICES=1 python PGGAN/train.py \
  --batch_size=64 \
  --mode='train' \
  --image_size=32 \
  --data_dir=/home/yhx/gan_lib/cifar-10 \
  --restore \
  --model='resnet' \
  --block_count=3



```









