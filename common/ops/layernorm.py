"""

"""

import tensorflow as tf


def Layernorm(name, norm_axes, inputs):
    """
    Args:
      name:
      norm_axes:
      inputs:

    Returns:
    """
    result = \
        tf.contrib.layers.layer_norm(inputs,
                                     center=True,
                                     scale=True,
                                     activation_fn=None,
                                     reuse=None,
                                     variables_collections=None,
                                     outputs_collections=None,
                                     trainable=True,
                                     begin_norm_axis=1,
                                     begin_params_axis=-1,
                                     scope=name)

    return result

# def Layernorm(name, norm_axes, inputs):
#     """
#     Args:
#       name:
#       norm_axes:
#       inputs:
#
#     Returns:
#     """
#     with tf.variable_scope('LayerNorm'):
#         mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)
#
#         # Assume the 'neurons' axis is the last of norm_axes.
#         # This is the case for fully-connected and BHWC conv layers.
#         n_neurons = inputs.get_shape().as_list()[norm_axes[-1]]
#
#         offset = tf.get_variable(name='offset', shape=[n_neurons, ], dtype=tf.float32,
#                                  initializer=tf.constant_initializer(0.))
#         scale = tf.get_variable(name='scale', shape=[n_neurons, ], dtype=tf.float32,
#                                 initializer=tf.constant_initializer(1.))
#
#         # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
#         offset = tf.reshape(offset, [1 for _ in range(len(norm_axes) - 1)] + [-1])
#         scale = tf.reshape(scale, [1 for _ in range(len(norm_axes) - 1)] + [-1])
#
#         result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)
#
#         return result
