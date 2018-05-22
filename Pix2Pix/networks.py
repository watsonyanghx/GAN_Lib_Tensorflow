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

import numpy as np
import common as lib
import common.ops.conv2d
import common.ops.linear
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


def resnet_generator(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding):
    """

    Args:
      input_images: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].
      outputs_channels:
      ngf:

    Returns:
      Returns generated image batch.
    """

    # (N, img_h, img_w, 1024)
    output = ResidualBlock(generator_inputs, generator_inputs.shape.as_list()[-1], ngf, 7, 'G.1',
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


def resnet_discriminator(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
                         conv_type, channel_multiplier, padding):
    """

    Args:
      input_images: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].
      outputs_channels:
      ngf:

    Returns:
      Returns generated image batch.
    """

    # (N, 4, 4, 1024)
    pass


def unet_generator(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding,
                   upsampe_method='depth_to_space'):
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
            if upsampe_method == 'resize':
                resized_input = tf.image.resize_images(rectified, [h * 2, w * 2],
                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            elif upsampe_method == 'depth_to_space':
                resized_input = tf.concat([rectified, rectified, rectified, rectified], axis=3)
                resized_input = tf.depth_to_space(resized_input, block_size=2)
            else:
                raise NotImplementedError('upsampe_method [%s] is not recognized' % upsampe_method)

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


# ######################## VGG ######################## #

VGG_MEAN = [103.939, 116.779, 123.68]


def vgg_generator(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding,
                  train_mode=None, trainable=None, vgg19_npy_path=None):
    """vgg generator

    Args:
      generator_inputs: A batch of images to translate.
        Images should be normalized already. Shape is [batch, height, width, channels].
      generator_outputs_channels:
      ngf:
      conv_type:
      channel_multiplier:
      padding:
      train_mode:
      trainable:

    Returns:
      Returns generated image batch.
    """
    if vgg19_npy_path is not None:
        data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
    else:
        data_dict = None

    rgb_scaled = (generator_inputs + 1) / 2  # [-1, 1] => [0, 1]
    rgb_scaled = rgb_scaled * 255.0

    print('\nrgb_scaled.shape.as_list(): {0}\n'.format(rgb_scaled.shape.as_list()))

    # Convert RGB to BGR
    red, green, blue = tf.split(value=rgb_scaled, num_or_size_splits=3, axis=3)
    # assert red.get_shape().as_list()[1:] == [224, 224, 1]
    # assert green.get_shape().as_list()[1:] == [224, 224, 1]
    # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ], axis=3)
    # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    conv1_1 = conv_layer(bgr, 3, 64, "conv1_1", trainable=False, data_dict=data_dict)
    conv1_2 = conv_layer(conv1_1, 64, 64, "conv1_2", trainable=False, data_dict=data_dict)
    pool1 = max_pool(conv1_2, 'pool1')  # [112, 112, 64], [256, 256, 64]

    conv2_1 = conv_layer(pool1, 64, 128, "conv2_1", trainable=False, data_dict=data_dict)
    conv2_2 = conv_layer(conv2_1, 128, 128, "conv2_2", trainable=False, data_dict=data_dict)
    pool2 = max_pool(conv2_2, 'pool2')  # [56, 56, 128], [128, 128, 128]

    conv3_1 = conv_layer(pool2, 128, 256, "conv3_1", trainable=False, data_dict=data_dict)
    conv3_2 = conv_layer(conv3_1, 256, 256, "conv3_2", trainable=False, data_dict=data_dict)
    conv3_3 = conv_layer(conv3_2, 256, 256, "conv3_3", trainable=False, data_dict=data_dict)
    conv3_4 = conv_layer(conv3_3, 256, 256, "conv3_4", trainable=False, data_dict=data_dict)
    pool3 = max_pool(conv3_4, 'pool3')  # [28, 28, 256], [64, 64, 256]

    conv4_1 = conv_layer(pool3, 256, 512, "conv4_1", trainable=False, data_dict=data_dict)
    conv4_2 = conv_layer(conv4_1, 512, 512, "conv4_2", trainable=False, data_dict=data_dict)
    conv4_3 = conv_layer(conv4_2, 512, 512, "conv4_3", trainable=False, data_dict=data_dict)
    conv4_4 = conv_layer(conv4_3, 512, 512, "conv4_4", trainable=False, data_dict=data_dict)
    pool4 = max_pool(conv4_4, 'pool4')  # [14, 14, 512], [32, 32, 512]

    conv5_1 = conv_layer(pool4, 512, 512, "conv5_1", trainable=True, data_dict=data_dict)
    conv5_2 = conv_layer(conv5_1, 512, 512, "conv5_2", trainable=True, data_dict=data_dict)
    conv5_3 = conv_layer(conv5_2, 512, 512, "conv5_3", trainable=True, data_dict=data_dict)
    conv5_4 = conv_layer(conv5_3, 512, 512, "conv5_4", trainable=True, data_dict=data_dict)
    pool5 = max_pool(conv5_4, 'pool5')  # [7, 7, 512], [16, 16, 512]

    # fc6 = fc_layer(pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    # relu6 = tf.nn.relu(fc6)
    # if train_mode is not None:
    #     relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(relu6, 0.5), lambda: relu6)
    # elif trainable:
    #     relu6 = tf.nn.dropout(relu6, 0.5)
    #
    # fc7 = fc_layer(relu6, 4096, 4096, "fc7")
    # relu7 = tf.nn.relu(fc7)
    # if train_mode is not None:
    #     relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(relu7, 0.5), lambda: relu7)
    # elif trainable:
    #     relu7 = tf.nn.dropout(relu7, 0.5)
    #
    # fc8 = fc_layer(relu7, 4096, 1000, "fc8")
    #
    # prob = tf.nn.softmax(fc8, name="prob")
    #
    # data_dict = None

    # ################ add other layers ################ #
    conv6_1 = lib.ops.conv2d.Conv2D(pool5, pool5.shape.as_list()[-1], 512, 3, 2, 'conv6_1',
                                    conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                    spectral_normed=False, update_collection=None, inputs_norm=False,
                                    he_init=True, biases=True)  # [8, 8, 512]

    conv6_2 = lib.ops.conv2d.Conv2D(conv6_1, conv6_1.shape.as_list()[-1], 512, 3, 2, 'conv6_2',
                                    conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                    spectral_normed=False, update_collection=None, inputs_norm=False,
                                    he_init=True, biases=True)  # [4, 4, 512]

    # decoder part
    conv6_2_decoder = tf.concat([conv6_2, conv6_2, conv6_2, conv6_2], axis=3, name="conv6_2_concat")
    conv6_2_decoder = tf.depth_to_space(conv6_2_decoder, 2, name='conv6_2_decoder')  # [8, 8, 512]
    conv6_2_decoder = lib.ops.conv2d.Conv2D(conv6_2_decoder, conv6_2_decoder.shape.as_list()[-1], 512, 1, 1,
                                            "conv6_2_conv",
                                            spectral_normed=False,
                                            update_collection=None,
                                            he_init=True, biases=True)

    conv6_2_decoder = tf.concat([conv6_2_decoder, conv6_1], axis=3)
    conv6_1_decoder = tf.concat([conv6_2_decoder, conv6_2_decoder, conv6_2_decoder, conv6_2_decoder], axis=3,
                                name="conv6_1_concat")
    conv6_1_decoder = tf.depth_to_space(conv6_1_decoder, 2, name='conv6_1_decoder')  # [16, 16, 512*2]
    conv6_1_decoder = lib.ops.conv2d.Conv2D(conv6_1_decoder, conv6_1_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv6_1_conv",
                                            spectral_normed=False,
                                            update_collection=None,
                                            he_init=True, biases=True)

    conv6_1_decoder = tf.concat([conv6_1_decoder, pool5], axis=3)
    pool5_decoder = tf.concat([conv6_1_decoder, conv6_1_decoder, conv6_1_decoder, conv6_1_decoder], axis=3,
                              name="pool5_decoder_concat")
    pool5_decoder = tf.depth_to_space(pool5_decoder, 2, name='pool5_decoder')  # [32, 32, 512*2]
    conv5_4_decoder = lib.ops.conv2d.Conv2D(pool5_decoder, pool5_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv5_4_decoder", he_init=True, biases=True)
    conv5_3_decoder = lib.ops.conv2d.Conv2D(conv5_4_decoder, conv5_4_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv5_3_decoder", he_init=True, biases=True)
    conv5_2_decoder = lib.ops.conv2d.Conv2D(conv5_3_decoder, conv5_3_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv5_2_decoder", he_init=True, biases=True)
    conv5_1_decoder = lib.ops.conv2d.Conv2D(conv5_2_decoder, conv5_2_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv5_1_decoder", he_init=True, biases=True)

    conv5_1_decoder = tf.concat([conv5_1_decoder, pool4], axis=3)
    pool4_decoder = tf.concat([conv5_1_decoder, conv5_1_decoder, conv5_1_decoder, conv5_1_decoder], axis=3,
                              name="pool4_decoder_concat")
    pool4_decoder = tf.depth_to_space(pool4_decoder, 2, name='pool4_decoder')  # [64, 64, 512*2]
    conv4_4_decoder = lib.ops.conv2d.Conv2D(pool4_decoder, pool4_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv4_4_decoder", he_init=True, biases=True)
    conv4_3_decoder = lib.ops.conv2d.Conv2D(conv4_4_decoder, conv4_4_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv4_3_decoder", he_init=True, biases=True)
    conv4_2_decoder = lib.ops.conv2d.Conv2D(conv4_3_decoder, conv4_3_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv4_2_decoder", he_init=True, biases=True)
    conv4_1_decoder = lib.ops.conv2d.Conv2D(conv4_2_decoder, conv4_2_decoder.shape.as_list()[-1], 256, 3, 1,
                                            "conv4_1_decoder", he_init=True, biases=True)

    conv4_1_decoder = tf.concat([conv4_1_decoder, pool3], axis=3)
    pool3_decoder = tf.concat([conv4_1_decoder, conv4_1_decoder, conv4_1_decoder, conv4_1_decoder], axis=3,
                              name="pool3_decoder_concat")
    pool3_decoder = tf.depth_to_space(pool3_decoder, 2, name='pool3_decoder')  # [128, 128, 256*2]
    conv3_4_decoder = lib.ops.conv2d.Conv2D(pool3_decoder, pool3_decoder.shape.as_list()[-1], 256, 3, 1,
                                            "conv3_4_decoder", he_init=True, biases=True)
    conv3_3_decoder = lib.ops.conv2d.Conv2D(conv3_4_decoder, conv3_4_decoder.shape.as_list()[-1], 256, 3, 1,
                                            "conv3_3_decoder", he_init=True, biases=True)
    conv3_2_decoder = lib.ops.conv2d.Conv2D(conv3_3_decoder, conv3_3_decoder.shape.as_list()[-1], 256, 3, 1,
                                            "conv3_2_decoder", he_init=True, biases=True)
    conv3_1_decoder = lib.ops.conv2d.Conv2D(conv3_2_decoder, conv3_2_decoder.shape.as_list()[-1], 128, 3, 1,
                                            "conv3_1_decoder", he_init=True, biases=True)

    conv3_1_decoder = tf.concat([conv3_1_decoder, pool2], axis=3)
    pool2_decoder = tf.concat([conv3_1_decoder, conv3_1_decoder, conv3_1_decoder, conv3_1_decoder], axis=3,
                              name="pool2_decoder_concat")
    pool2_decoder = tf.depth_to_space(pool2_decoder, 2, name='pool2_decoder')  # [256, 256, 128*2]
    conv2_2_decoder = lib.ops.conv2d.Conv2D(pool2_decoder, pool2_decoder.shape.as_list()[-1], 128, 3, 1,
                                            "conv2_2_decoder", he_init=True, biases=True)
    conv2_1_decoder = lib.ops.conv2d.Conv2D(conv2_2_decoder, conv2_2_decoder.shape.as_list()[-1], 64, 3, 1,
                                            "conv2_1_decoder", he_init=True, biases=True)

    conv2_1_decoder = tf.concat([conv2_1_decoder, pool1], axis=3)
    pool1_decoder = tf.concat([conv2_1_decoder, conv2_1_decoder, conv2_1_decoder, conv2_1_decoder], axis=3,
                              name="pool1_decoder_concat")
    pool1_decoder = tf.depth_to_space(pool1_decoder, 2, name='pool1_decoder')  # [512, 512, 64*2]
    conv1_2_decoder = lib.ops.conv2d.Conv2D(pool1_decoder, pool1_decoder.shape.as_list()[-1], 64, 3, 1,
                                            "conv1_2_decoder", he_init=True, biases=True)

    conv1_2_decoder = tf.concat([conv1_2_decoder, bgr], axis=3)
    bgr_output = lib.ops.conv2d.Conv2D(conv1_2_decoder, conv1_2_decoder.shape.as_list()[-1], 3, 3, 1,
                                       "bgr_output", he_init=True, biases=True)

    bgr_output = tf.nn.tanh(bgr_output)

    # convert bgr to rgb
    b, g, r = tf.split(bgr_output, 3, axis=3)
    rgb = tf.concat([r, g, b], axis=3)

    return rgb


def vgg_discriminator(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
                      conv_type, channel_multiplier, padding):
    """vgg generator

    Args:
      discrim_inputs: A batch of images to translate.
        Images should be normalized already. Shape is [batch, height, width, channels].
      discrim_targets:
      ndf:
      spectral_normed:
      update_collection:
      conv_type:
      channel_multiplier:
      padding:

    Returns:
      Returns generated image batch.
    """

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # block 1
    output = lib.ops.conv2d.Conv2D(inputs, inputs.shape.as_list()[-1], 32, 1, 1, 'D_conv1',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [512, 512, 32]
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv2',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [512, 512, 64]
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # [256, 256, 64]

    # block 2
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv3',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [256, 256, 64]
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv4',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [256, 256, 64]
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # [128, 128, 64]

    # block 3
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv5',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [128, 128, 64]
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv6',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [128, 128, 64]
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # [64, 64, 64]

    # fc layer
    output = tf.reshape(output, [output.shape.as_list()[0], -1])
    output = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 100, 'D.fc1',
                                   spectral_normed=spectral_normed, update_collection=update_collection,
                                   inputs_norm=False,
                                   biases=True, initialization=None, weightnorm=None, gain=1.)
    output = tf.nn.tanh(output)

    output = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 2, 'D.fc2',
                                   spectral_normed=spectral_normed, update_collection=update_collection,
                                   inputs_norm=False,
                                   biases=True, initialization=None, weightnorm=None, gain=1.)
    output = tf.nn.tanh(output)

    output = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 1, 'D.Output',
                                   spectral_normed=spectral_normed, update_collection=update_collection,
                                   inputs_norm=False,
                                   biases=True, initialization=None, weightnorm=None, gain=1.)
    # output = tf.nn.sigmoid(output)

    return output


# ######################## helper function for VGG ######################## #
def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, in_channels, out_channels, name, trainable=True, data_dict=None):
    # with tf.variable_scope(name):
    filt, conv_biases = get_conv_var(3, in_channels, out_channels, name,
                                     trainable=trainable, data_dict=data_dict)

    conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, conv_biases)
    relu = tf.nn.relu(bias)

    return relu


def fc_layer(bottom, in_size, out_size, name, trainable=True, data_dict=None):
    # with tf.variable_scope(name):
    weights, biases = get_fc_var(in_size, out_size, name, trainable=trainable, data_dict=data_dict)

    x = tf.reshape(bottom, [-1, in_size])
    fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    return fc


def get_conv_var(filter_size, in_channels, out_channels, name, trainable=True, data_dict=None):
    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    filters = get_var(initial_value, name, 0, name + "_filters", trainable=trainable, data_dict=data_dict)

    initial_value = tf.truncated_normal([out_channels], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases", trainable=trainable, data_dict=data_dict)

    return filters, biases


def get_fc_var(in_size, out_size, name, trainable=True, data_dict=None):
    initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
    weights = get_var(initial_value, name, 0, name + "_weights", trainable=trainable, data_dict=data_dict)

    initial_value = tf.truncated_normal([out_size], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases", trainable=trainable, data_dict=data_dict)

    return weights, biases


def get_var(initial_value, name, idx, var_name, trainable=True, data_dict=None):
    """vgg generator

    Args:
      initial_value: A batch of images to translate.
        Images should be normalized already. Shape is [batch, height, width, channels].
      idx:
      var_name:
      trainable:
      data_dict: Pre-trained variables.

    Returns:
      Returns generated image batch.
    """
    if data_dict is not None and name in data_dict:
        value = data_dict[name][idx]
    else:
        value = initial_value

    if trainable:
        var = tf.Variable(value, name=var_name)
    else:
        var = tf.constant(value, dtype=tf.float32, name=var_name)

    # var_dict[(name, idx)] = var

    # print var_name, var.get_shape().as_list()
    assert var.get_shape() == initial_value.get_shape()

    return var

# def save_npy(sess, npy_path="./vgg19-save.npy"):
#     assert isinstance(sess, tf.Session)
#
#     data_dict = {}
#
#     for (name, idx), var in list(var_dict.items()):
#         var_out = sess.run(var)
#         if name not in data_dict:
#             data_dict[name] = {}
#         data_dict[name][idx] = var_out
#
#     np.save(npy_path, data_dict)
#     print(("file saved", npy_path))
#     return npy_path
#
#
# def get_var_count():
#     count = 0
#     for v in list(var_dict.values()):
#         count += reduce(lambda x, y: x * y, v.get_shape().as_list())
#     return count
