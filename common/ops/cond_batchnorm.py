"""

"""

import tensorflow as tf


def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True,
              labels=None, n_labels=None):
    """Conditional Batchnorm (dumoulin et al 2016) for BHWC conv filtermaps
    Args:
      name:
      axes:
      inputs: Tensor of shape (batch size, height, width, num_channels)
      is_training:
      stats_iter:
      update_moving_stats:
      fused:
      labels:
      n_labels:

    Returns:
    """
    with tf.variable_scope('CondBatchNorm'):
        if axes != [0, 1, 2]:
            raise Exception('Axes is not supported in Conditional BatchNorm!')

        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        shape = mean.get_shape().as_list()  # shape is [1, 1, 1, n]
        offset_m = tf.get_variable(name='offset', shape=[n_labels, shape[3]], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.))
        scale_m = tf.get_variable(name='scale', shape=[n_labels, shape[3]], dtype=tf.float32,
                                  initializer=tf.constant_initializer(1.))

        offset = tf.nn.embedding_lookup(offset_m, labels)
        scale = tf.nn.embedding_lookup(scale_m, labels)

        result = tf.nn.batch_normalization(inputs, mean, var, offset[:, None, None, :], scale[:, None, None, :], 1e-5)

        return result
