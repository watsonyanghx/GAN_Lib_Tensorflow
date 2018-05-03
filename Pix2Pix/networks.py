"""
Networks for GAN Pix2Pix.

1. Instance Normalization in G only.
2. Hinge loss.
3. Spectral Normalization in D only.

"""

# import os
# import sys
#
# sys.path.append(os.getcwd())

import common as lib
import common.ops.conv2d
import common.ops.normalization

from common.resnet_block import *


def norm_layer(inputs, decay=0.9, epsilon=1e-5, is_training=True, norm_type="BN"):
    """

    Args:
      inputs: A batch of images to be normed. Shape is [batch, height, width, channels].
      epsilon: Default value: 1e-5 for BatchNorm, 1e-6 for InstanceNorm.
      norm_type: "BN" for BatchNorm, "IN" for InstanceNorm.

    Returns:
      Returns generated image batch.
    """
    if norm_type == "BN":
        outputs = lib.ops.normalization.batch_norm(inputs, decay=decay, epsilon=epsilon, is_training=True)
    elif norm_type == "IN":
        outputs = lib.ops.normalization.instance_norm(inputs, epsilon=epsilon)
    else:
        raise NotImplementedError('Normalization [%s] is not implemented!' % norm_type)

    return outputs


def resnet_generator(input_images, outputs_channels, ngf=64):
    """

    Args:
      input_images: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].
      outputs_channels:
      ngf:

    Returns:
      Returns generated image batch.
    """

    # with tf.variable_scope('g_net'):
    # (N, img_h, img_w, 1024)
    output = ResidualBlock(input_images, input_images.shape.as_list()[-1], ngf, 7, 'G.1',
                           resample='None', labels=None, activation_fn='relu')
    output = norm_layer(output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="BN")
    output = tf.nn.relu(output)

    # (N, img_h // 2, img_w // 2, 1024)
    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2 ** i
        # model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
        #                     stride=2, padding=1, bias=use_bias)
        output = \
            lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], ngf * mult * 2, filter_size=3, stride=2,
                                  name='Conv2D.%d' % (i + 2), conv_type='conv2d', channel_multiplier=0, padding='SAME',
                                  spectral_normed=False, update_collection=None, inputs_norm=False, he_init=True,
                                  mask_type=None, weightnorm=None, biases=True, gain=1.)
        output = ResidualBlock(output, output.shape.as_list()[-1], ngf * mult * 2, 3, 'G.%d' % (i + 2),
                               resample='down', labels=None, activation_fn='relu')
        output = norm_layer(output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="BN")
        output = tf.nn.relu(output)

    # (N, img_h, img_w, 1024)
    output = ResidualBlock(output, output.shape.as_list()[-1], 256, 3, 'G.3',
                           resample='down', labels=None, activation_fn='relu')

    output = Normalize('G.OutputN', output)
    output = nonlinearity(output, activation_fn='relu')
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 3, 3, 1, 'G.Output',
                                   he_init=False, biases=True)
    output = tf.nn.tanh(output)

    return output


def resnet_discriminator(input_images, outputs_channels, ngf=64):
    """

    Args:
      input_images: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].
      outputs_channels:
      ngf:

    Returns:
      Returns generated image batch.
    """

    # with tf.variable_scope('g_net'):
    # (N, 4, 4, 1024)
    pass


def unet_generator(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        # output = gen_conv(generator_inputs, args.ngf)
        output = lib.ops.conv2d.Conv2D(generator_inputs, generator_inputs.shape.as_list()[-1], ngf, 4, 2, 'Conv2D',
                                       conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
                                       spectral_normed=False, update_collection=None, inputs_norm=False,
                                       he_init=True, biases=True)
        layers.append(output)

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = nonlinearity(layers[-1], 'lrelu', 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            # convolved = gen_conv(rectified, out_channels)
            convolved = lib.ops.conv2d.Conv2D(rectified, rectified.shape.as_list()[-1], out_channels, 4, 2, 'Conv2D',
                                              conv_type=conv_type, channel_multiplier=channel_multiplier,
                                              padding=padding,
                                              spectral_normed=False, update_collection=None, inputs_norm=False,
                                              he_init=True, biases=True)
            # output = batchnorm(convolved)
            output = norm_layer(convolved, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            # output = convolved
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                inputs = layers[-1]
            else:
                inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(inputs)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            # output = gen_deconv(rectified, out_channels)
            _b, h, w, _c = rectified.shape
            resized_input = tf.image.resize_images(rectified, [h * 2, w * 2],
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = lib.ops.conv2d.Conv2D(resized_input, resized_input.shape.as_list()[-1], out_channels, 4, 1,
                                           'Conv2D',
                                           conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
                                           spectral_normed=False, update_collection=None, inputs_norm=False,
                                           he_init=True, biases=True)
            # output = tf.layers.conv2d_transpose(rectified, out_channels, kernel_size=4, strides=(2, 2),
            #                                     padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer)
            # output = batchnorm(output)

            output = norm_layer(output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        inputs = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(inputs)
        # output = gen_deconv(rectified, generator_outputs_channels)
        _b, h, w, _c = rectified.shape
        resized_input = tf.image.resize_images(rectified, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        output = lib.ops.conv2d.Conv2D(resized_input, resized_input.shape.as_list()[-1], generator_outputs_channels, 4,
                                       1, 'Conv2D',
                                       conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
                                       spectral_normed=False, update_collection=None, inputs_norm=False,
                                       he_init=True, biases=True)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def unet_discriminator(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
                       conv_type, channel_multiplier, padding):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        # convolved = discrim_conv(inputs, ndf, stride=2)
        padded_input = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], ndf, 4, 2,
                                          'Conv2D',
                                          conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
                                          spectral_normed=spectral_normed,
                                          update_collection=update_collection,
                                          inputs_norm=False,
                                          he_init=True, biases=True)

        rectified = nonlinearity(convolved, 'lrelu', 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels_ = ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            # convolved = discrim_conv(layers[-1], out_channels_, stride=stride)
            padded_input = tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], out_channels_, 4, stride,
                                              'Conv2D',
                                              conv_type=conv_type, channel_multiplier=channel_multiplier,
                                              padding=padding,
                                              spectral_normed=spectral_normed,
                                              update_collection=update_collection,
                                              inputs_norm=False,
                                              he_init=True, biases=True)

            # normalized = batchnorm(convolved)
            # normalized = norm_layer(convolved, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            normalized = convolved
            rectified = nonlinearity(normalized, 'lrelu', 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        # convolved = discrim_conv(rectified, out_channels=1, stride=1)
        padded_input = tf.pad(rectified, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], 1, 4, 1,
                                          'Conv2D',
                                          conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
                                          spectral_normed=spectral_normed,
                                          update_collection=update_collection,
                                          inputs_norm=False,
                                          he_init=True, biases=True)

        # output = tf.sigmoid(convolved)
        output = convolved
        layers.append(output)

    return layers[-1]
