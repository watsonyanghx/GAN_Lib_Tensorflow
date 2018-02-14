# from https://github.com/chainer/chainerrl/blob/f119a1fe210dd31ea123d244258d9b5edc21fba4/chainerrl/misc/copy_param.py

import numpy as np
import tensorflow as tf
import os
import sys
import subprocess
import scipy.misc
from scipy.misc import imsave


def record_setting(out):
    """Record scripts and commandline arguments"""
    out = out.split()[0].strip()
    if not os.path.exists(out):
        os.mkdir(out)
    subprocess.call("cp *.py %s" % out, shell=True)

    with open(out + "/command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")


# Image grid saver, based on color_grid_vis from github.com/Newmu
def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.float):
        X = (255.99 * X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, int(n_samples / rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        # X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w] = x

    imsave(save_path, img)


def get_z(batchsize, n_hidden=128):
    """Get random noise 'z'.

    Args:
      batchsize:
      n_hidden:

    Returns:
    """
    z = np.random.normal(size=(batchsize, n_hidden)).astype(np.float32)
    z /= np.sqrt(np.sum(z * z, axis=1, keepdims=True) / n_hidden + 1e-8)
    return z


# ------


def scope_has_variables(scope):
    return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0


def optimistic_restore(session, save_file):
    """

    Args:
      session:
      save_file:

    Returns:
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []

    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

    # print('\n--------variables stored:--------')
    # for var_name, saved_var_name in var_names:
    #     print(var_name)

    print('\n--------variables to restore:--------')
    for var in restore_vars:
        print(var)


def get_loss(disc_real, disc_fake, loss_type='HINGE'):
    if loss_type == 'HINGE':
        disc_real_l = tf.reduce_mean(tf.nn.relu(1. - disc_real))
        disc_fake_l = tf.reduce_mean(tf.nn.relu(1. + disc_fake))
        d_loss = disc_real_l + disc_fake_l

        g_loss = -tf.reduce_mean(disc_fake)
    elif loss_type == 'WGAN':
        disc_real_l = - tf.reduce_mean(disc_real)
        disc_fake_l = tf.reduce_mean(disc_fake)
        d_loss = disc_real_l + disc_fake_l

        # clip_d_vars_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for var in d_vars]
        # # Paste the code bellow to where `session.run(d_train_op)`
        # session.run(clip_d_vars_op)

        g_loss = -tf.reduce_mean(disc_fake)
    elif loss_type == 'WGAN-GP':
        disc_real_l = - tf.reduce_mean(disc_real)
        disc_fake_l = tf.reduce_mean(disc_fake)
        d_loss = disc_real_l + disc_fake_l

        # Paste the code bellow where `get_loss()` is called.
        # # Gradient Penalty
        # alpha = tf.random_uniform(shape=[args.batch_size, 1, 1, 1], minval=0., maxval=1.)
        # differences = fake_data - real_data
        # interpolates = real_data + (alpha * differences)
        # gradients = tf.gradients(
        #     model.get_discriminator(interpolates, real_labels, 'NO_OPS', reuse=True)[0], [interpolates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        # gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)
        # d_loss_gan += gradient_penalty

        g_loss = -tf.reduce_mean(disc_fake)
    elif loss_type == 'LSGAN':
        disc_real_l = tf.reduce_mean(tf.square(1. - disc_real))
        disc_fake_l = tf.reduce_mean(tf.square(disc_fake))
        d_loss = (disc_real_l + disc_fake_l) / 2

        g_loss = tf.reduce_mean(tf.square(1. - disc_fake)) / 2
    elif loss_type == 'CGAN':
        disc_real_l = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                    labels=tf.ones_like(disc_real)))
        disc_fake_l = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                    labels=tf.zeros_like(disc_fake)))
        d_loss = disc_real_l + disc_fake_l

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                    labels=tf.ones_like(disc_fake)))
    elif loss_type == 'Goodfellow':
        disc_real_l = -tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_real)))
        disc_fake_l = -tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(disc_fake)))
        d_loss = disc_real_l + disc_fake_l

        g_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_fake)))

    return d_loss, g_loss
