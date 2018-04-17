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

parser = argparse.ArgumentParser(description='Train script')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--image_size', type=int, default=4)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--out', type=str, default='./result', help='Directory to output the result.')
parser.add_argument('--display_interval', type=int, default=100,
                    help='Interval of displaying log to console.')
parser.add_argument('--evaluation_interval', type=int, default=10000,
                    help='Interval of evaluation inception score.')
parser.add_argument('--out_image_interval', type=int, default=1000,
                    help='Interval of out image.')
parser.add_argument('--n_dis', type=int, default=5,
                    help='Number of discriminator update per generator update.')

#
parser.add_argument('--data_dir', type=str, default='./cifar-10')
parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint',
                    help='Directory to stroe checkpoints and summaries.')
parser.add_argument('--z_dim', type=int, default=512, help='dim')
parser.add_argument('--image_dim', type=int, default=3072,
                    help='image dimension after flatting image')
parser.add_argument('--restore', action="store_true",
                    help='If true, restore variables from checkpoint.')

# PPGAN
parser.add_argument('--model', type=str, default="resnet", help='resnet, nvidia')
parser.add_argument('--block_count', type=int, default=0,
                    help='[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]:'
                         '[4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]')
parser.add_argument('--trans', action="store_true", help='default: False')
parser.add_argument('--inputs_norm', action="store_true", help='default: False')

args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def train():
    if args.model == 'nvidia':
        from PGGAN.model_nvidia import PGGAN
    elif args.model == 'resnet':
        from PGGAN.model_resnet import PGGAN
    else:
        raise NotImplementedError('Not supported model!')

    # lib.misc.record_setting(args.out)
    # report_keys = ["stage", "loss_dis", "loss_gen", "g", "inception_mean", "inception_std", "FID"]

    # Function for reading data
    train_gen, dev_gen = lib.data.cifar10.load(args.batch_size, args.data_dir)

    def inf_train_gen():
        while True:
            for images_, labels_ in train_gen():
                yield images_, labels_

    gen = inf_train_gen()

    # placeholder
    alpha = tf.placeholder(tf.float32, shape=None, name='alpha')
    real_data_int = tf.placeholder(tf.int32, [args.batch_size, args.image_dim], name='images')
    real_data = 2 * (tf.cast(real_data_int, tf.float32) / 256. - 0.5)
    real_data += tf.random_uniform(shape=[args.batch_size, args.image_dim], minval=0., maxval=1. / 128)
    real_data = tf.transpose(tf.reshape(real_data, shape=[-1, 3, 32, 32]), perm=[0, 2, 3, 1], name='NCHW_to_NHWC')

    if args.trans and args.block_count:
        real_data = tf.image.resize_images(real_data, [args.image_size // 2, args.image_size // 2])
        real_data = tf.image.resize_images(real_data, [args.image_size, args.image_size])
    else:
        real_data = tf.image.resize_images(real_data, [args.image_size, args.image_size])
    print('real_data.shape: {}'.format(real_data.shape.as_list()))
    # all_real_labels = tf.placeholder(tf.int32, [args.batch_size], name='labels')

    #
    # with tf.variable_scope('PGGAN_Model'):
    model = PGGAN(args)

    disc_real = model.get_discriminator(real_data, alpha, update_collection=None)

    z = tf.random_normal([args.batch_size, args.z_dim])
    x_fake = model.get_generator(z, alpha)
    disc_fake = model.get_discriminator(x_fake, alpha, update_collection='NO_OPS', reuse=True)

    # with tf.variable_scope('Optimization'):
    disc_real_l = tf.reduce_mean(tf.nn.relu(1. - disc_real))
    disc_fake_l = tf.reduce_mean(tf.nn.relu(1. + disc_fake))
    d_loss = disc_real_l + disc_fake_l
    g_loss = -tf.reduce_mean(disc_fake)
    tf.summary.scalar('d_loss', d_loss)
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
    # learning_rate = tf.train.polynomial_decay(0.0002, global_step, args.max_iter // 2 * args.n_dis, 0.0001)
    # tf.summary.scalar('global_step', global_step)
    # tf.summary.scalar('learning_rate', learning_rate)
    # g_opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0., beta2=0.99, epsilon=1e-8)
    g_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0., beta2=0.9, epsilon=1e-8)
    g_train_op = g_opt.minimize(loss=g_loss, var_list=g_vars, global_step=global_step)
    # d_opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0., beta2=0.99, epsilon=1e-8)
    d_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0., beta2=0.9, epsilon=1e-8)
    d_train_op = d_opt.minimize(loss=d_loss, var_list=d_vars, global_step=global_step)

    # Function for generating samples
    fixed_z = tf.constant(lib.misc.get_z(100, n_hidden=args.z_dim), dtype=tf.float32)
    # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    # fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
    fixed_noise_samples = model.get_generator(fixed_z, alpha, reuse=True)

    def generate_image(session, frame, alpha_):
        samples = session.run(fixed_noise_samples, feed_dict={alpha: alpha_})
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        # lib.misc.save_images(samples.reshape((100, 32, 32, 3)), 'samples_{}.png'.format(frame))
        lib.misc.save_images(samples, 'samples_{}.png'.format(frame))

    # Function for calculating inception score
    random_z = tf.random_normal([100, args.z_dim])
    # fake_labels_100 = tf.cast(tf.random_uniform([100]) * 10, tf.int32)
    samples_100 = model.get_generator(random_z, alpha, reuse=True)

    def get_inception_score(session, n, alpha_):
        all_samples = []
        for i in range(int(n / 100)):
            all_samples.append(session.run(samples_100, feed_dict={alpha: alpha_}))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
        # all_samples = all_samples.reshape((-1, 32, 32, 3))
        return lib.inception.inception_score.get_inception_score(list(all_samples))

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
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
            g_loss_, _ = sess.run([g_loss, g_train_op], feed_dict={alpha: (step * 1.0) / args.max_iter})

            for _ in range(args.n_dis):
                _data, _labels = next(gen)
                summaries, d_loss_, _ = sess.run([summary_op, d_loss, d_train_op],
                                                 feed_dict={alpha: (step * 1.0) / args.max_iter,
                                                            real_data_int: _data})
            summary_writer.add_summary(summaries, global_step=step)

            if step % args.display_interval == args.display_interval - 1:
                tf.logging.info('step: {}, g_loss: {}, d_loss: {}'.format(step, g_loss_, d_loss_))

            if step % args.evaluation_interval == args.evaluation_interval - 1:
                inception_score = get_inception_score(sess, 50000, (step * 1.0) / args.max_iter)
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
                                              feed_dict={alpha: (step * 1.0) / args.max_iter,
                                                         real_data_int: images})
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

                generate_image(sess, step, (step * 1.0) / args.max_iter)

                if not os.path.exists(args.checkpoint_dir):
                    os.mkdir(args.checkpoint_dir)
                saver.save(sess, os.path.join(args.checkpoint_dir, 'model.ckpt'), global_step=step)

            lib.plot.tick()

        summary_writer.flush()
        summary_writer.close()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
