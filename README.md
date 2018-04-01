# GAN_Lib_Tensorflow

Tensorflow implemention of various GAN.

Please refer to corresponding folder for more details.


## Prerequisite

- Python 3.5.4
- TensorFlow 1.5
- Numpy
- Scipy


## Quantitative evaluation

| Method | [Inception](https://arxiv.org/abs/1606.03498) | Inception (Official) | [FID](https://arxiv.org/abs/1706.08500) |
| ------------- | ------------- | ------------- | ------------- |
| Real data  | 12.0 | 11.24 | 3.2 (train vs test) |
| [PGGAN](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/tree/master/PGGAN)  | - | 8.80 ± 0.05 (-, Unsupervised) | - |
| [SNGAN](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/tree/master/SNGAN)  | 8.43 ± 0.12 (ResNet, Supervised) | 8.24 ± 0.08 (ResNet, Unsupervised) | - |
| [ACGAN](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/tree/master/ACGAN)  |  (ResNet, Supervised) |  8.25 ± 0.07 (-, Unsupervised) | - |


Inception scores are calculated by average of 10 evaluation with 5000 samples.

#### TODO

- FID

- MS-SSIM


## Generated images

- **SNGAN (Spectral Normalization for Generative Adversarial Networks)**

    ![sample](https://github.com/watsonyanghx/GAN_Lib_Tensorflow/blob/master/SNGAN/img/samples_99999.png)



