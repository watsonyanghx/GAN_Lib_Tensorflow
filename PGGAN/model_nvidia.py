"""
PGGAN based on Nvidia architecture.
"""
# -*- coding: utf-8 -*-

# import numpy as np
import tensorflow as tf

import common as lib
import common.ops.conv2d
import common.ops.linear
import common.ops.pixelnorm


def lrelu(x, leakiness=0.2):
    assert leakiness <= 1, "leakiness must be <= 1"
    return tf.maximum(x, leakiness * x)


def minibatch_std(x):
    # x: (B, H, W, C)
    m = tf.reduce_mean(x, axis=0, keepdims=True)  # (1, H, W, C)
    a = tf.tile(m, multiples=[x.shape.as_list()[0], 1, 1, 1])  # (B, H, W, C)
    v = tf.reduce_mean((x - a) * (x - a), axis=0, keepdims=True)  # (1, H, W, C)
    std = tf.reduce_mean(tf.sqrt(v + 1e-8), keepdims=True)  # (1, 1, 1, 1)
    std = tf.tile(std, (x.shape.as_list()[0], x.shape.as_list()[1], x.shape.as_list()[2], 1))

    return tf.concat([x, std], axis=3)


class PGGAN(object):
    def __init__(self, args):
        """
        Args:
          args:
        """
        self.bc = args.block_count  # Count of up/down block.
        self.trans = args.trans  # If trans.
        self.inputs_norm = args.inputs_norm

    def get_dim(self, stage):
        """
        Args:
          stage:
        Return:
        """
        return min(2048 / (2 ** stage), 512)

    def generator_block(self, inputs, out_dim, name='generator_block'):
        """
        Args:
          inputs:
          out_dim:
          name:
        Return:
        """
        with tf.variable_scope(name):
            output = tf.concat(values=[inputs, inputs, inputs, inputs], axis=3)
            output = tf.depth_to_space(output, block_size=2)

            output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], out_dim, 3, 1, 'Conv.1',
                                           inputs_norm=self.inputs_norm, he_init=True, biases=True)
            output = lib.ops.pixelnorm.Pixelnorm(output)
            output = lrelu(output)

            output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], out_dim, 3, 1, 'Conv.2',
                                           inputs_norm=self.inputs_norm, he_init=True, biases=True)
            output = lib.ops.pixelnorm.Pixelnorm(output)
            output = lrelu(output)

        return output

    def get_generator(self, z_var, alpha, training=True, reuse=False):
        """g-net
        Args:
          z_var:
          alpha:
          training:
          reuse:
        Return:
        """
        with tf.variable_scope('g_net', reuse=reuse):
            z_var_ = tf.reshape(z_var, [z_var.shape.as_list()[0], -1])

            # (N, 4, 4, 512)
            output = lib.ops.linear.Linear(z_var_, z_var_.shape.as_list()[-1], 4 * 4 * 512, 'G.Input',
                                           inputs_norm=self.inputs_norm)
            output = tf.reshape(output, [-1, 4, 4, 512])
            output = lib.ops.pixelnorm.Pixelnorm(output)
            output = lrelu(output)
            print('G.Input: {}'.format(output.shape.as_list()))
            # (N, 4, 4, 512)
            output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 512, 3, 1, 'G.Conv',
                                           inputs_norm=self.inputs_norm, he_init=True, biases=True)
            output = lib.ops.pixelnorm.Pixelnorm(output)
            output = lrelu(output)
            print('G.Conv: {}'.format(output.shape.as_list()))

            for i in range(self.bc - 1):
                output = self.generator_block(output, self.get_dim(i), 'G.UpBlock.{}'.format(i + 1))
                print('G.UpBlock.{}: {}'.format(i + 1, output.shape.as_list()))

            if self.trans:
                toRGB1 = self.generator_block(output, self.get_dim(self.bc - 1), 'G.UpBlock.{}'.format(self.bc))
                print('G.UpBlock.{}: {}'.format(self.bc, toRGB1.shape.as_list()))
                toRGB1 = lib.ops.conv2d.Conv2D(toRGB1, toRGB1.shape.as_list()[-1], 3, 1, 1, 'G.{}_toRGB1'.format(self.bc),
                                               inputs_norm=self.inputs_norm, he_init=True, biases=True)
                print('G.{}_toRGB1: {}'.format(self.bc, toRGB1.shape.as_list()))

                # skip connection
                toRGB2 = tf.concat(values=[output, output, output, output], axis=3)
                toRGB2 = tf.depth_to_space(input=toRGB2, block_size=2)
                toRGB2 = lib.ops.conv2d.Conv2D(toRGB2, toRGB2.shape.as_list()[-1], 3, 1, 1, 'G.{}_toRGB2'.format(self.bc),
                                               inputs_norm=self.inputs_norm, he_init=True, biases=True)
                print('G.{}_toRGB2: {}'.format(self.bc, toRGB2.shape.as_list()))

                # fade in
                toRGB = (1 - alpha) * toRGB2 + alpha * toRGB1
            else:
                if self.bc > 0:
                    toRGB = self.generator_block(output, self.get_dim(self.bc - 1), 'G.UpBlock.{}'.format(self.bc))
                    print('G.UpBlock.{}: {}'.format(self.bc, toRGB.shape.as_list()))
                else:
                    toRGB = output
                toRGB = lib.ops.conv2d.Conv2D(toRGB, toRGB.shape.as_list()[-1], 3, 1, 1, 'G.{}_toRGB'.format(self.bc),
                                              inputs_norm=self.inputs_norm, he_init=True, biases=True)
                print('G.{}_toRGB: {}'.format(self.bc, toRGB.shape.as_list()))

        return toRGB

    def discriminator_block(self, inputs, out_dim, name, spectral_normed=False, update_collection=None, reuse=False):
        """
        Args:
          inputs:
          out_dim:
          name:
          spectral_normed:
          update_collection:
          reuse:
        Return:
        """
        with tf.variable_scope(name):
            output = lib.ops.conv2d.Conv2D(inputs, inputs.shape.as_list()[-1], inputs.shape.as_list()[-1], 3, 1,
                                           'Conv.1',
                                           spectral_normed=spectral_normed,
                                           update_collection=update_collection,
                                           reuse=reuse,
                                           he_init=True, biases=True)
            # output = custom_ops.Normalize('Normalize.1', output)
            output = lrelu(output)

            output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], out_dim, 3, 1, 'Conv.2',
                                           spectral_normed=spectral_normed,
                                           update_collection=update_collection,
                                           reuse=reuse,
                                           he_init=True, biases=True)
            # output = custom_ops.Normalize('Normalize.2', output)
            output = lrelu(output)

            output = tf.nn.avg_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        return output

    def get_discriminator(self, x_var, alpha, spectral_normed=True, update_collection=None, reuse=False):
        """d-net
        Args:
          x_var:
          alpha:
          spectral_normed:
          update_collection:
          reuse:
        Return:
        """
        with tf.variable_scope('d_net', reuse=reuse):
            if self.trans:
                fromRGB1 = lib.ops.conv2d.Conv2D(x_var, x_var.shape.as_list()[-1], self.get_dim(self.bc - 1), 1, 1,
                                                 'D.{}_fromRGB1'.format(self.bc),
                                                 spectral_normed=spectral_normed,
                                                 update_collection=update_collection,
                                                 reuse=reuse,
                                                 he_init=True, biases=True)
                print('D.{}_fromRGB1: {}'.format(self.bc, fromRGB1.shape.as_list()))
                fromRGB1 = self.discriminator_block(fromRGB1, self.get_dim(self.bc - 1), 'D.Block.{}'.format(self.bc),
                                                    spectral_normed=spectral_normed,
                                                    update_collection=update_collection,
                                                    reuse=reuse)
                print('D.Block.{}: {}'.format(self.bc, fromRGB1.shape.as_list()))

                # skip connection
                fromRGB2 = tf.nn.avg_pool(x_var, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                fromRGB2 = lib.ops.conv2d.Conv2D(fromRGB2, fromRGB2.shape.as_list()[-1], self.get_dim(self.bc - 1), 1,
                                                 1, 'D.{}_fromRGB2'.format(self.bc),
                                                 spectral_normed=spectral_normed,
                                                 update_collection=update_collection,
                                                 reuse=reuse,
                                                 he_init=True, biases=True)
                print('D.{}_fromRGB2: {}'.format(self.bc, fromRGB2.shape.as_list()))

                # fade in
                x_code = (1 - alpha) * fromRGB2 + alpha * fromRGB1
            else:
                x_code = lib.ops.conv2d.Conv2D(x_var, x_var.shape.as_list()[-1], self.get_dim(self.bc - 1), 1, 1,
                                               'D.{}_fromRGB'.format(self.bc),
                                               spectral_normed=spectral_normed,
                                               update_collection=update_collection,
                                               reuse=reuse,
                                               he_init=True, biases=True)
                print('D.{}_fromRGB: {}'.format(self.bc, x_code.shape.as_list()))
                if self.bc > 0:
                    x_code = self.discriminator_block(x_code, self.get_dim(self.bc - 1), 'D.Block.{}'.format(self.bc),
                                                      spectral_normed=spectral_normed,
                                                      update_collection=update_collection,
                                                      reuse=reuse)
                    print('D.Block.{}: {}'.format(self.bc, x_code.shape.as_list()))

            for i in range(1, self.bc):
                x_code = self.discriminator_block(x_code, self.get_dim(self.bc - 1 - i),
                                                  'D.Block.{}'.format(self.bc - i),
                                                  spectral_normed=spectral_normed,
                                                  update_collection=update_collection,
                                                  reuse=reuse)
                print('D.Block.{}: {}'.format(self.bc - i, x_code.shape.as_list()))

            output = minibatch_std(x_code)
            output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], self.get_dim(self.bc - 1), 3, 1,
                                           'D.Conv',
                                           spectral_normed=spectral_normed,
                                           update_collection=update_collection,
                                           reuse=reuse,
                                           he_init=True, biases=True)
            output = lrelu(output)

            output = tf.reduce_mean(output, axis=[1, 2])
            logits = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 1, 'D.Output')
            logits = tf.reshape(logits, [-1])

        return logits
