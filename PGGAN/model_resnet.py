"""
PGGAN based on ResNet architecture.
"""
# -*- coding: utf-8 -*-

# import numpy as np
import tensorflow as tf

# import common as lib
import common.ops.embedding
import common.resnet_block


class PGGAN(object):
    def __init__(self, args):
        """
        Args:
          args: .
        """
        self.bc = args.block_count  # Count of up/down block.
        self.trans = args.trans  # If trans.
        self.inputs_norm = args.inputs_norm

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
            output_tensor = common.resnet_block.Generator_PGGAN(z_var_, self.bc, self.trans, alpha, self.inputs_norm,
                                                                training=training)

            return output_tensor

    def get_discriminator(self, x_var, alpha, labels=None, update_collection=None, reuse=False):
        """d-net
        Args:
          x_var:
          labels:
          alpha:
          update_collection:
          reuse:
        Return:
        """
        with tf.variable_scope('d_net', reuse=reuse):
            # if c_var is not None:
            #     c_code = c_var
            #
            #     # embedding labels, and concatenate to 'output'.
            #     # (N, EMBEDDING_DIM)
            #     embedding_y = lib.ops.embedding.embed_y(labels, VOCAB_SIZE, EMBEDDING_DIM, word2vec_file=WORD2VEC_FILE)
            #     embedding_y = lib.ops.linear.Linear('Discriminator.Embedding_y', EMBEDDING_DIM, DIM_D, embedding_y,
            #                                         spectral_normed=True,
            #                                         update_collection=update_collection,
            #                                         reuse=reuse,
            #                                         biases=True)  # (N, DIM_D)
            # else:
            #     c_code = None

            c_code = labels
            logits = common.resnet_block.Discriminator_PGGAN(x_var, c_code, self.bc, self.trans, alpha,
                                                             self.inputs_norm,
                                                             update_collection=update_collection,
                                                             reuse=reuse)
        return logits
