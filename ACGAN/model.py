"""

"""

# import numpy as np
# import tensorflow as tf

import common as lib
import common.ops.conv2d
import common.ops.linear

from common.resnet_block import *


class ACGAN(object):
    def __init__(self):
        """
        Args:
        """

    def get_generator(self, z_var, labels=None, training=True, reuse=False):
        """g-net
        Args:
          z_var:
          labels:
          training:
          reuse:
        Return:
        """
        with tf.variable_scope('g_net', reuse=reuse):
            z_var_ = tf.reshape(z_var, [z_var.shape.as_list()[0], -1])

            # (N, 4, 4, 1024)
            output = lib.ops.linear.Linear(z_var_, z_var_.shape.as_list()[-1], 4 * 4 * 1024, 'G.Input')
            output = tf.reshape(output, [-1, 4, 4, 1024])

            output = ResidualBlock(output, output.shape.as_list()[-1], 256, 3, 'G.1',
                                   resample='up', labels=labels, activation_fn='relu')
            output = ResidualBlock(output, output.shape.as_list()[-1], 256, 3, 'G.2',
                                   resample='up', labels=labels, activation_fn='relu')
            output = ResidualBlock(output, output.shape.as_list()[-1], 256, 3, 'G.3',
                                   resample='up', labels=labels, activation_fn='relu')
            output = Normalize('G.OutputN', output)
            output = nonlinearity(output, activation_fn='relu')
            output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 3, 3, 1, 'G.Output',
                                           he_init=False, biases=True)
            output = tf.nn.tanh(output)

            return output

    def get_discriminator(self, x_var, labels=None, update_collection=None, reuse=False):
        """d-net
        Args:
          x_var:
          labels:
          update_collection:
          reuse:
        Return:
        """
        with tf.variable_scope('d_net', reuse=reuse):
            output = OptimizedResBlockDisc1(x_var, activation_fn='lrelu')
            output = ResidualBlock(output, output.shape.as_list()[-1], 128, 3, 'D.DownBlock.2',
                                   spectral_normed=False,
                                   update_collection=update_collection,
                                   resample='down', labels=labels, biases=True, activation_fn='lrelu')
            output = ResidualBlock(output, output.shape.as_list()[-1], 128, 3, 'D.NoneBlock.3',
                                   spectral_normed=False,
                                   update_collection=update_collection,
                                   resample=None, labels=labels, biases=True, activation_fn='lrelu')
            output = ResidualBlock(output, output.shape.as_list()[-1], 128, 3, 'D.NoneBlock.4',
                                   spectral_normed=False,
                                   update_collection=update_collection,
                                   resample=None, labels=labels, biases=True, activation_fn='lrelu')
            output = nonlinearity(output, activation_fn='lrelu')
            output = tf.reduce_mean(output, axis=[1, 2])

            # GAN
            output_wgan = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 1, 'D.Output',
                                                spectral_normed=False,
                                                update_collection=update_collection,
                                                biases=True)
            output_wgan = tf.reshape(output_wgan, [-1])

            # Auxiliary Classifier
            output_acgan = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 10, 'D.ACGANOutput',
                                                 spectral_normed=False,
                                                 update_collection=update_collection,
                                                 biases=True)

            return output_wgan, output_acgan
