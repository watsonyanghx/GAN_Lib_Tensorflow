"""

"""
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import sys

sys.path.append(os.getcwd())

# from progressive.evaluation import sample_generate, sample_generate_light, calc_inception, calc_FID

import common as lib
import common.misc
import common.data.cifar10
import common.inception.inception_score
import common.plot

from ACGAN.model import ACGAN

parser = argparse.ArgumentParser(description='Train script')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mode', type=str, default='train')
# parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--loss_type', type=str, default='HINGE')
parser.add_argument('--n_dis', type=int, default=5,
                    help='Number of discriminator update per generator update.')
parser.add_argument('--acgan_scale_G', type=float, default=0.1,
                    help='acgan_scale_G.')
# parser.add_argument('--out', type=str, default='./result', help='Directory to output the result.')
parser.add_argument('--display_interval', type=int, default=100,
                    help='Interval of displaying log to console.')
parser.add_argument('--evaluation_interval', type=int, default=1000,
                    help='Interval of evaluation inception score.')
parser.add_argument('--out_image_interval', type=int, default=1000,
                    help='Interval of out image.')

#
parser.add_argument('--data_dir', type=str, default='/home/yhx/sn_gan/cifar-10')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                    help='Directory to stroe checkpoints and summaries.')
parser.add_argument('--z_dim', type=int, default=128, help='dim')
parser.add_argument('--image_dim', type=int, default=3072,
                    help='image dimension after flatting image')
parser.add_argument('--restore', action="store_true",
                    help='If true, restore variables from checkpoint.')

# PPGAN
# parser.add_argument('--model', type=str, default="resnet", help='resnet, nvidia')
# parser.add_argument('--block_count', type=int, default=0,
#                     help='[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]:'
#                          '[4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]')
# parser.add_argument('--trans', action="store_true", help='default: False')
# parser.add_argument('--inputs_norm', action="store_true", help='default: False')

args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def train():
    # Function for reading data
    train_gen, dev_gen = lib.data.cifar10.load(args.batch_size, args.data_dir)

    def inf_train_gen():
        while True:
            for images_, labels_ in train_gen():
                yield images_, labels_

    gen = inf_train_gen()

    # placeholder
    real_data_int = tf.placeholder(tf.int32, [args.batch_size, args.image_dim], name='images')
    real_data = 2 * (tf.cast(real_data_int, tf.float32) / 256. - 0.5)
    real_data += tf.random_uniform(shape=[args.batch_size, args.image_dim], minval=0., maxval=1. / 128)
    real_data = tf.transpose(tf.reshape(real_data, shape=[-1, 3, 32, 32]), perm=[0, 2, 3, 1], name='NCHW_to_NHWC')
    print('\nreal_data.shape: {}'.format(real_data.shape.as_list()))
    real_labels = tf.placeholder(tf.int32, [args.batch_size], name='labels')

    #
    # with tf.variable_scope('PGGAN_Model'):
    model = ACGAN()

    disc_real, disc_real_acgan = model.get_discriminator(real_data, real_labels, update_collection=None)

    z = tf.random_normal([args.batch_size, args.z_dim])
    fake_labels = tf.cast(tf.random_uniform([args.batch_size]) * 10, tf.int32)
    x_fake = model.get_generator(z, fake_labels)
    disc_fake, disc_fake_acgan = model.get_discriminator(x_fake, fake_labels, update_collection='NO_OPS', reuse=True)

    # with tf.variable_scope('Optimization'):
    d_loss_gan, g_loss_gan = lib.misc.get_loss(disc_real, disc_fake, loss_type=args.loss_type)

    # gradient penalty
    alpha = tf.random_uniform(shape=[args.batch_size, 1, 1, 1], minval=0., maxval=1.)
    differences = x_fake - real_data
    interpolates = real_data + (alpha * differences)
    gradients = tf.gradients(
        model.get_discriminator(interpolates, real_labels, 'NO_OPS', reuse=True)[0], [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]) + 1e-10)
    gradient_penalty = 10 * tf.reduce_mean(tf.square((slopes - 1.)))
    d_loss_gan += gradient_penalty

    tf.summary.scalar('d_loss_gan', d_loss_gan)
    tf.summary.scalar('g_loss_gan', g_loss_gan)
    d_loss_acgan = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_real_acgan, labels=real_labels)
    )
    d_loss = d_loss_gan + d_loss_acgan
    tf.summary.scalar('d_loss_acgan', d_loss_acgan)
    tf.summary.scalar('d_loss', d_loss)

    g_loss_acgan = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
    )
    g_loss = g_loss_gan + args.acgan_scale_G * g_loss_acgan
    tf.summary.scalar('g_loss_acgan', g_loss_acgan)
    tf.summary.scalar('g_loss', g_loss)

    g_vars = [var for var in tf.trainable_variables() if 'g_net' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'd_net' in var.name]
    print('\n--------g_vars:--------')
    for var in g_vars:
        print(var.name)
    print('\n--------d_vars:--------')
    for var in d_vars:
        print(var.name)
    print('\n--------tf.trainable_variables():--------')
    for var in tf.trainable_variables():
        print(var)

    global_step = tf.Variable(0, trainable=False)
    tf.summary.scalar('global_step', global_step)
    learning_rate = tf.train.polynomial_decay(0.0004, global_step, args.max_iter // 2, 0.0002)
    tf.summary.scalar('learning_rate', learning_rate)
    # g_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999, epsilon=1e-8)
    g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0., beta2=0.9, epsilon=1e-8)
    g_train_op = g_opt.minimize(loss=g_loss, var_list=g_vars, global_step=global_step)
    # d_opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.999, epsilon=1e-8)
    d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0., beta2=0.9, epsilon=1e-8)
    d_train_op = d_opt.minimize(loss=d_loss, var_list=d_vars)

    # Function for generating samples
    fixed_z = tf.constant(lib.misc.get_z(100, n_hidden=args.z_dim), dtype=tf.float32)
    # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
    fixed_noise_samples = model.get_generator(fixed_z, fixed_labels, reuse=True)

    def generate_image(session, frame):
        samples = session.run(fixed_noise_samples)
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        lib.misc.save_images(samples, 'samples_{}.png'.format(frame))

    # Function for calculating inception score
    random_z = tf.random_normal([100, args.z_dim])
    fake_labels_100 = tf.cast(tf.random_uniform([100]) * 10, tf.int32)
    samples_100 = model.get_generator(random_z, fake_labels_100, reuse=True)

    def get_inception_score(session, n):
        all_samples = []
        for i in range(int(n / 100)):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
        # all_samples = all_samples.reshape((-1, 32, 32, 3))
        return lib.inception.inception_score.get_inception_score(list(all_samples))

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(args.checkpoint_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        if args.restore:
            ckpt = tf.train.latest_checkpoint(args.checkpoint_dir)
            if ckpt:
                print('Restore model from: {}...'.format(ckpt))
                lib.misc.optimistic_restore(sess, ckpt)
            else:
                print('No checkpoint found in: {}'.format(args.checkpoint_dir))

        # Run the training
        for step in range(0, args.max_iter):
            if step > 0:
                g_loss_gan_, g_loss_acgan_, _ = sess.run([g_loss_gan, g_loss_acgan, g_train_op])

            for i_ in range(args.n_dis):
                _data, _labels = next(gen)

                summaries, d_loss_gan_, d_loss_acgan_, _ = \
                    sess.run([summary_op, d_loss_gan, d_loss_acgan, d_train_op],
                             feed_dict={real_labels: _labels,
                                        real_data_int: _data})
            summary_writer.add_summary(summaries, global_step=step)

            if step % args.display_interval == args.display_interval - 1:
                tf.logging.info('step: {}, g_loss_gan: {}, g_loss_acgan: {}, d_loss_gan: {}, d_loss_acgan: {}'
                                .format(step, g_loss_gan_, g_loss_acgan_, d_loss_gan_, d_loss_acgan_))

            if step % args.evaluation_interval == args.evaluation_interval - 1:
                inception_score = get_inception_score(sess, 50000)
                lib.plot.plot('inception_50k', inception_score[0])
                lib.plot.plot('inception_50k_std', inception_score[1])
                lib.plot.flush()

                # tf.logging.info('step: {}, inception_50k: {}, inception_50k_std: {}'.format(
                #     step, inception_score[0], inception_score[1]))

            # Calculate dev loss and generate samples every 100 iters
            if step % args.out_image_interval == args.out_image_interval - 1:
                dev_disc_costs = []
                for images, _labels in dev_gen():
                    _dev_disc_cost = sess.run(d_loss,
                                              feed_dict={real_labels: _labels,
                                                         real_data_int: images})
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

                generate_image(sess, step)

                if not os.path.exists(args.checkpoint_dir):
                    os.mkdir(args.checkpoint_dir)
                saver.save(sess, os.path.join(args.checkpoint_dir, 'model.ckpt'), global_step=step)

            lib.plot.tick()

        summary_writer.flush()
        summary_writer.close()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
