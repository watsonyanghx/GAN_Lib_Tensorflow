"""

"""

# import numpy as np
import tensorflow as tf


def Pixelnorm(inputs, eps=1e-8):
    """From PGGAN.
    Args:
      inputs: (B, H, W, C)
      eps:

    Returns:
    """
    print('Using Pixelnorm...')
    # outputs = inputs / tf.sqrt(tf.reduce_mean(inputs ** 2, axis=3, keepdims=True) + eps)

    alpha = 1.0 / tf.sqrt(tf.reduce_mean(inputs * inputs, axis=3, keepdims=True) + eps)  # (B, H, W, 1)
    alpha = tf.tile(alpha, multiples=[1, 1, 1, inputs.shape.as_list()[3]])
    outputs = alpha * inputs

    return outputs
