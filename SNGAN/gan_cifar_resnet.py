"""SNGAN ResNet for conditional generation of CIFAR-10"""

# no ACGAN, 1
# NoLabelConcatInG, 1
# DECAY, 1
# N_CRITIC = 5
# biases=True

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

import time
import functools
import locale

import common.misc
import common.data.cifar10
import common.inception.inception_score

import common as lib
import common.ops.linear
import common.ops.cond_batchnorm
import common.ops.conv2d
import common.ops.embedding
import common.ops.batchnorm
import common.ops.layernorm
import common.plot

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the extracted files here!
DATA_DIR = '/home/yhx/sn_gan/cifar-10'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

BATCH_SIZE = 64  # Critic batch size
GEN_BS_MULTIPLE = 2  # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 100000  # How many iterations to train for
DIM_G = 128  # Generator dimensionality
DIM_D = 128  # Critic dimensionality
NORMALIZATION_G = True  # Use batchnorm in generator?
NORMALIZATION_D = False  # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (32*32*3)
LR = 0.0002  # 2e-4  # Initial learning rate
DECAY = True  # Whether to decay LR over learning
N_CRITIC = 5  # 5  # Critic steps per generator steps
INCEPTION_FREQUENCY = 1000  # How frequently to calculate Inception score

CONDITIONAL = True  # Whether to train a conditional or unconditional model
ACGAN = False  # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1  # How to scale generator's ACGAN loss relative to WGAN loss

# SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"
# WORD2VEC_FILE = np.load(os.path.join(DATA_DIR, 'glove_y.npy')).astype('float32')
WORD2VEC_FILE = None
VOCAB_SIZE = 10
EMBEDDING_DIM = 300  # 620
CHECKPOINT_DIR = './checkpoint'
LOSS_TYPE = 'HINGE'  # 'Goodfellow', 'HINGE', 'WGAN', 'WGAN-GP'
SOFT_PLUS = False
RESTORE = False
CONCAT_LABEL = False  # whether concat label to 'z' in Generator.

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print("WARNING! Conditional model without normalization in D might be effectively unconditional!")

N_GPUS = 1
if N_GPUS not in [1, 2]:
    raise Exception('Only 1 or 2 GPUs supported!')
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
if len(DEVICES) == 1:  # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())


def nonlinearity(x, activation_fn='relu', leakiness=0.2):
    if activation_fn == 'relu':
        return tf.nn.relu(x)
    if activation_fn == 'lrelu':
        assert 0 < leakiness <= 1, "leakiness must be <= 1"
        return tf.maximum(x, leakiness * x)


def Normalize(name, inputs, labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""

    with tf.variable_scope(name):
        if not CONDITIONAL:
            labels = None
        if CONDITIONAL and ACGAN and ('D.' in name):
            labels = None

        if ('D.' in name) and NORMALIZATION_D:
            return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
        elif ('G.' in name) and NORMALIZATION_G:
            if labels is not None:
                outputs = lib.ops.cond_batchnorm.Batchnorm(name, [0, 1, 2], inputs, labels=labels, n_labels=10)
                return outputs
            else:
                outputs = lib.ops.batchnorm.Batchnorm(inputs, fused=True)
                return outputs
        else:
            return inputs


def ConvMeanPool(inputs, output_dim, filter_size=3, stride=1, name=None,
                 spectral_normed=False, update_collection=None, inputs_norm=False,
                 he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(inputs, inputs.shape.as_list()[-1], output_dim, filter_size, stride, name,
                                   spectral_normed=spectral_normed,
                                   update_collection=update_collection,
                                   he_init=he_init, biases=biases)
    # output = tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    output = tf.add_n(
        [output[:, ::2, ::2, :], output[:, 1::2, ::2, :], output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
    return output


def MeanPoolConv(inputs, output_dim, filter_size=3, stride=1, name=None,
                 spectral_normed=False, update_collection=None, inputs_norm=False,
                 he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, ::2, ::2, :], output[:, 1::2, ::2, :], output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
    # output = tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], output_dim, filter_size, stride, name,
                                   spectral_normed=spectral_normed,
                                   update_collection=update_collection,
                                   he_init=he_init, biases=biases)

    return output


def UpsampleConv(inputs, output_dim, filter_size=3, stride=1, name=None,
                 spectral_normed=False, update_collection=None, inputs_norm=False,
                 he_init=False, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=3)
    output = tf.depth_to_space(output, 2)
    # w, h = inputs.shape.as_list()[1], inputs.shape.as_list()[2]
    # output = tf.image.resize_images(inputs, [w * 2, h * 2])
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], output_dim, filter_size, stride, name,
                                   spectral_normed=spectral_normed,
                                   update_collection=update_collection,
                                   he_init=he_init, biases=biases)

    return output


def ResidualBlock(inputs, input_dim, output_dim, filter_size, name,
                  spectral_normed=False, update_collection=None, inputs_norm=False,
                  resample=None, labels=None, biases=True):
    """resample: None, 'down', or 'up'.
    """
    if resample == 'down':
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample == 'up':
        conv_1 = functools.partial(UpsampleConv, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample is None:
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim)
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample is None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(inputs=inputs, output_dim=output_dim, filter_size=1, name=name + '.Shortcut',
                                 spectral_normed=spectral_normed,
                                 update_collection=update_collection,
                                 he_init=False, biases=biases)

    output = inputs
    output = Normalize(name + '.N1', output, labels=labels)
    output = nonlinearity(output)
    # if resample == 'up':
    #     output = nonlinearity(output)
    # else:
    #     output = lrelu(output, leakiness=0.2)

    output = conv_1(inputs=output, filter_size=filter_size, name=name + '.Conv1',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    he_init=True, biases=biases)

    output = Normalize(name + '.N2', output, labels=labels)
    output = nonlinearity(output)
    # if resample == 'up':
    #     output = nonlinearity(output)
    # else:
    #     output = lrelu(output, leakiness=0.2)

    output = conv_2(inputs=output, filter_size=filter_size, name=name + '.Conv2',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    he_init=True, biases=biases)

    return shortcut + output


def OptimizedResBlockDisc1(inputs,
                           spectral_normed=False, update_collection=None, inputs_norm=False,
                           biases=True):
    conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2 = functools.partial(ConvMeanPool, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut(inputs=inputs, output_dim=DIM_D, filter_size=1, name='D.Block.1.Shortcut',
                             spectral_normed=spectral_normed,
                             update_collection=update_collection,
                             he_init=False, biases=biases)

    output = inputs
    output = conv_1(inputs=output, filter_size=3, name='D.Block.1.Conv1',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    he_init=True, biases=biases)
    output = nonlinearity(output)
    # output = lrelu(output, leakiness=0.2)
    output = conv_2(inputs=output, filter_size=3, name='D.Block.1.Conv2',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    he_init=True, biases=biases)
    return shortcut + output


def Generator(n_samples_, labels, noise=None, reuse=False):
    with tf.variable_scope("Generator", reuse=reuse):
        if noise is None:
            noise = tf.random_normal([n_samples_, 128])

        if CONCAT_LABEL:
            # embedding labels, and concatenate to 'noise'.
            # (N, EMBEDDING_DIM)
            embedding_y = lib.ops.embedding.embed_y(labels, VOCAB_SIZE, EMBEDDING_DIM, word2vec_file=WORD2VEC_FILE)
            # (N, DIM_D)
            embedding_y = lib.ops.linear.Linear(embedding_y, EMBEDDING_DIM, DIM_D, 'G.Embedding_y', biases=True)
            noise = tf.concat(values=[noise, embedding_y], axis=1)
            output = lib.ops.linear.Linear(noise, 128 * 2, 4 * 4 * DIM_G * 8, 'G.Input')
        else:
            output = lib.ops.linear.Linear(noise, 128, 4 * 4 * DIM_G * 8, 'G.Input')

        output = tf.reshape(output, [-1, 4, 4, DIM_G * 8])
        output = ResidualBlock(output, DIM_G * 8, DIM_G * 2, 3, 'G.Block.1', resample='up', labels=labels, biases=True)
        output = ResidualBlock(output, DIM_G * 2, DIM_G * 2, 3, 'G.Block.2', resample='up', labels=labels, biases=True)
        output = ResidualBlock(output, DIM_G * 2, DIM_G * 2, 3, 'G.Block.3', resample='up', labels=labels, biases=True)
        output = Normalize('G.OutputNorm', output, labels)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(output, DIM_G * 2, 3, 3, 1, 'G.Output', he_init=False)
        output = tf.tanh(output)
        # return tf.reshape(tf.transpose(output, [0, 3, 1, 2], name='NHWC_to_NCHW'), [-1, OUTPUT_DIM])
        return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs, labels, update_collection=None, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        output = tf.reshape(inputs, [-1, 32, 32, 3])
        output = OptimizedResBlockDisc1(output,
                                        spectral_normed=True,
                                        update_collection=update_collection,
                                        biases=True)

        # embedding labels, and concatenate to 'output'.
        # (N, EMBEDDING_DIM)
        embedding_y = lib.ops.embedding.embed_y(labels, VOCAB_SIZE, EMBEDDING_DIM, word2vec_file=WORD2VEC_FILE)
        embedding_y = lib.ops.linear.Linear(embedding_y, EMBEDDING_DIM, DIM_D, 'D.Embedding_y',
                                            spectral_normed=True,
                                            update_collection=update_collection,
                                            biases=True)  # (N, DIM_D)

        embedding_y = tf.expand_dims(tf.expand_dims(embedding_y, axis=1), axis=1)
        embedding_y = tf.tile(embedding_y, multiples=[1, output.shape.as_list()[1], output.shape.as_list()[2], 1])
        output = tf.concat(values=[output, embedding_y], axis=3)

        output = ResidualBlock(output, DIM_D * 2, DIM_D, 3, 'D.Block.2',
                               spectral_normed=True,
                               update_collection=update_collection,
                               resample='down', labels=labels, biases=True)

        output = ResidualBlock(output, DIM_D, DIM_D, 3, 'D.Block.3',
                               spectral_normed=True,
                               update_collection=update_collection,
                               resample=None, labels=labels, biases=True)
        output = ResidualBlock(output, DIM_D, DIM_D, 3, 'D.Block.4',
                               spectral_normed=True,
                               update_collection=update_collection,
                               resample=None, labels=labels, biases=True)
        output = nonlinearity(output)
        # output = lrelu(output, leakiness=0.2)
        output = tf.reduce_mean(output, axis=[1, 2])
        output_wgan = lib.ops.linear.Linear(output, DIM_D, 1, 'D.Output',
                                            spectral_normed=True,
                                            update_collection=update_collection)
        output_wgan = tf.reshape(output_wgan, [-1])
        if CONDITIONAL and ACGAN:
            output_acgan = lib.ops.linear.Linear(output, DIM_D, 10, 'D.ACGANOutput',
                                                 spectral_normed=True,
                                                 update_collection=update_collection,
                                                 biases=True)
            return output_wgan, output_acgan
        else:
            return output_wgan, None


# with tf.Graph().as_default() as g:
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    fake_data_splits = []
    for i, device in enumerate(DEVICES):
        with tf.device(device):
            if i > 0:
                fake_data_splits.append(Generator(int(BATCH_SIZE / len(DEVICES)), labels_splits[i], reuse=True))
            else:
                fake_data_splits.append(Generator(int(BATCH_SIZE / len(DEVICES)), labels_splits[i]))

    all_real_data = tf.reshape(2 * ((tf.cast(all_real_data_int, tf.float32) / 256.) - .5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE, OUTPUT_DIM], minval=0., maxval=1. / 128)  # dequantize
    all_real_data = tf.reshape(
        tf.transpose(tf.reshape(all_real_data, [-1, 3, 32, 32]), perm=[0, 2, 3, 1]), [-1, OUTPUT_DIM])
    all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

    DEVICES_A = DEVICES[int(len(DEVICES) / 2):]
    # DEVICES_B = DEVICES[:int(len(DEVICES) / 2)]

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    for i, device in enumerate(DEVICES_A):
        with tf.device(device):
            real_and_fake_data = tf.concat(values=[
                all_real_data_splits[i],
                all_real_data_splits[len(DEVICES_A) + i],
                fake_data_splits[i],
                fake_data_splits[len(DEVICES_A) + i]
            ], axis=0)
            real_and_fake_labels = tf.concat(values=[
                labels_splits[i],
                labels_splits[len(DEVICES_A) + i],
                labels_splits[i],
                labels_splits[len(DEVICES_A) + i]
            ], axis=0)
            disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels, update_collection=None)
            disc_real = disc_all[:int(BATCH_SIZE / len(DEVICES_A))]
            disc_fake = disc_all[int(BATCH_SIZE / len(DEVICES_A)):]
            if LOSS_TYPE == 'Goodfellow':
                if SOFT_PLUS:
                    disc_real_l = -tf.reduce_mean(tf.nn.softplus(tf.log(tf.nn.sigmoid(disc_real))))
                    disc_fake_l = -tf.reduce_mean(tf.nn.softplus(tf.log(1 - tf.nn.sigmoid(disc_fake))))
                else:
                    disc_real_l = -tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_real)))
                    disc_fake_l = -tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(disc_fake)))
                disc_costs.append(disc_real_l + disc_fake_l)
            elif LOSS_TYPE == 'HINGE':
                if SOFT_PLUS:
                    disc_real_l = tf.reduce_mean(tf.nn.softplus(-tf.minimum(0., -1 + disc_real)))
                    disc_fake_l = tf.reduce_mean(tf.nn.softplus(-tf.minimum(0., -1 - disc_fake)))
                else:
                    # disc_real_l = -tf.reduce_mean(tf.minimum(0., -1 + disc_real))
                    # disc_fake_l = -tf.reduce_mean(tf.minimum(0., -1 - disc_fake))
                    disc_real_l = tf.reduce_mean(tf.nn.relu(1. - disc_real))
                    disc_fake_l = tf.reduce_mean(tf.nn.relu(1. + disc_fake))
                disc_costs.append(disc_real_l + disc_fake_l)
            elif LOSS_TYPE == 'WGAN':
                if SOFT_PLUS:
                    disc_costs.append(
                        tf.reduce_mean(tf.nn.softplus(disc_fake)) + tf.reduce_mean(tf.nn.softplus(-disc_real)))
                else:
                    disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))

            if CONDITIONAL and ACGAN:
                disc_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=disc_all_acgan[:int(BATCH_SIZE / len(DEVICES_A))],
                        labels=real_and_fake_labels[:int(BATCH_SIZE / len(DEVICES_A))])
                ))
                disc_acgan_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[:int(BATCH_SIZE / len(DEVICES_A))], axis=1)),
                            real_and_fake_labels[:int(BATCH_SIZE / len(DEVICES_A))]
                        ),
                        tf.float32
                    )
                ))
                disc_acgan_fake_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[int(BATCH_SIZE / len(DEVICES_A)):], axis=1)),
                            real_and_fake_labels[int(BATCH_SIZE / len(DEVICES_A)):]
                        ),
                        tf.float32
                    )
                ))

    # gradient_penalty, not included
    # if LOSS_TYPE == 'WGAN-GP'
    # for i, device in enumerate(DEVICES_B):
    #     with tf.device(device):
    #         real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A) + i]], axis=0)
    #         fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A) + i]], axis=0)
    #         labels = tf.concat([
    #             labels_splits[i],
    #             labels_splits[len(DEVICES_A) + i],
    #         ], axis=0)
    #         alpha = tf.random_uniform(
    #             shape=[int(BATCH_SIZE / len(DEVICES_A)), 1],
    #             minval=0.,
    #             maxval=1.
    #         )
    #         differences = fake_data - real_data
    #         interpolates = real_data + (alpha * differences)
    #         gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
    #         slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    #         gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)
    #         disc_costs.append(gradient_penalty)

    disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
    tf.summary.scalar('D_wgan_cost', disc_wgan)
    if CONDITIONAL and ACGAN:
        disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
        disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
        disc_cost = disc_wgan + (ACGAN_SCALE * disc_acgan)

        tf.summary.scalar('D_acgan_cost', disc_acgan)
        tf.summary.scalar('D_acgan_accuracy', disc_acgan_acc)
        tf.summary.scalar('D_acgan_fake_accuracy', disc_acgan_fake_acc)
        tf.summary.scalar('D_cost', disc_cost)
    else:
        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan

    if DECAY:
        decay = tf.where(
            tf.less(_iteration, 50000),
            tf.maximum(0., 1. - (tf.cast(_iteration, tf.float32) / 100000)), 0.5)
    else:
        decay = 1.
    tf.summary.scalar('lr', LR * decay)

    gen_costs = []
    gen_acgan_costs = []
    for device in DEVICES:
        with tf.device(device):
            n_samples = GEN_BS_MULTIPLE * int(BATCH_SIZE / len(DEVICES))
            fake_labels = tf.cast(tf.random_uniform([n_samples]) * 10, tf.int32)
            if CONDITIONAL and ACGAN:
                disc_fake, disc_fake_acgan = Discriminator(Generator(n_samples, fake_labels, reuse=True),
                                                           fake_labels,
                                                           update_collection="NO_OPS",
                                                           reuse=True)
                gen_costs.append(-tf.reduce_mean(tf.nn.softplus(disc_fake)))
                # gen_costs.append(-tf.reduce_mean(disc_fake))
                gen_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                ))
            else:
                disc_fake, _ = Discriminator(Generator(n_samples, fake_labels, reuse=True),
                                             fake_labels,
                                             update_collection="NO_OPS",
                                             reuse=True)
                if LOSS_TYPE == 'Goodfellow':
                    if SOFT_PLUS:
                        gen_costs.append(tf.reduce_mean(tf.nn.softplus(-tf.log(tf.nn.sigmoid(disc_fake)))))
                    else:
                        gen_costs.append(-tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_fake))))
                elif LOSS_TYPE == 'HINGE':
                    if SOFT_PLUS:
                        gen_costs.append(tf.reduce_mean(tf.nn.softplus(-disc_fake)))
                    else:
                        gen_costs.append(-tf.reduce_mean(disc_fake))
                elif LOSS_TYPE == 'WGAN':
                    if SOFT_PLUS:
                        gen_costs.append(tf.reduce_mean(tf.nn.softplus(-disc_fake)))
                    else:
                        gen_costs.append(-tf.reduce_mean(disc_fake))
    gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
    tf.summary.scalar('G_wgan_cost', gen_cost)
    if CONDITIONAL and ACGAN:
        gen_cost += (ACGAN_SCALE_G * (tf.add_n(gen_acgan_costs) / len(DEVICES)))
        tf.summary.scalar('G_acgan_costs', tf.add_n(gen_acgan_costs) / len(DEVICES))
        tf.summary.scalar('G_cost', gen_cost)

    # gen_params = lib.params_with_name('Generator')
    # disc_params = lib.params_with_name('D.')
    gen_params = [var for var in tf.trainable_variables() if 'Generator' in var.name]
    print('\ngen_params:')
    for var in gen_params:
        print(var.name)

    disc_params = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
    print('\ndisc_params:')
    for var in disc_params:
        print(var.name)

    print('\ntrainable_variables.name:')
    for var in tf.trainable_variables():
        print(var.name)

    gen_opt = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=gen_params)
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Function for generating samples
    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
    fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise, reuse=True)


    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        common.misc.save_images(samples.reshape((100, 32, 32, 3)), 'samples_{}.png'.format(frame))


    # Function for calculating inception score
    fake_labels_100 = tf.cast(tf.random_uniform([100]) * 10, tf.int32)
    samples_100 = Generator(100, fake_labels_100, reuse=True)


    def get_inception_score(n):
        all_samples = []
        for i in range(int(n / 100)):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
        # all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
        all_samples = all_samples.reshape((-1, 32, 32, 3))
        return common.inception.inception_score.get_inception_score(list(all_samples))


    # Function for reading data
    train_gen, dev_gen = common.data.cifar10.load(BATCH_SIZE, DATA_DIR)


    def inf_train_gen():
        while True:
            for images_, labels_ in train_gen():
                yield images_, labels_


    gen = inf_train_gen()

    for name, grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print("{} Params:".format(name))
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g is None:
                print("\t{} ({}) [no grad!]".format(v.name, shape_str))
            else:
                print("\t{} ({})".format(v.name, shape_str))
        print("Total param count: {}".format(locale.format("%d", total_param_count, grouping=True)))

    summaries_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)
    summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR, graph=session.graph)
    session.run(tf.global_variables_initializer())

    if RESTORE:
        ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if ckpt:
            print('restore model from: {}...'.format(ckpt))
            saver.restore(session, ckpt)

    for iteration in range(ITERS):
        start_time = time.time()

        if 0 < iteration:
            _ = session.run([gen_train_op], feed_dict={_iteration: iteration})

        for i in range(N_CRITIC):
            _data, _labels = next(gen)
            if CONDITIONAL and ACGAN:
                _disc_cost, _disc_wgan, _gen_cost, _disc_acgan, _disc_acgan_acc, \
                _disc_acgan_fake_acc, _, summaries = session.run(
                    [disc_cost, disc_wgan, gen_cost, disc_acgan, disc_acgan_acc,
                     disc_acgan_fake_acc, disc_train_op, summaries_op],
                    feed_dict={all_real_data_int: _data,
                               all_real_labels: _labels,
                               _iteration: iteration})
            else:
                _disc_cost, _disc_wgan, _gen_cost, _, summaries = session.run(
                    [disc_cost, disc_wgan, gen_cost, disc_train_op, summaries_op],
                    feed_dict={all_real_data_int: _data,
                               all_real_labels: _labels,
                               _iteration: iteration})

        summary_writer.add_summary(summaries, global_step=iteration)

        # lib.plot.plot('cost', _disc_cost)
        lib.plot.plot('d_cost', _disc_wgan)
        lib.plot.plot('g_cost', _gen_cost)
        if CONDITIONAL and ACGAN:
            lib.plot.plot('disc_wgan', _disc_wgan)
            lib.plot.plot('acgan', _disc_acgan)
            lib.plot.plot('acc_real', _disc_acgan_acc)
            lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
        # lib.plot.plot('time', time.time() - start_time)

        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY - 1:
            inception_score = get_inception_score(50000)
            lib.plot.plot('inception_50k', inception_score[0])
            lib.plot.plot('inception_50k_std', inception_score[1])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _labels in dev_gen():
                _dev_disc_cost = session.run([disc_cost],
                                             feed_dict={all_real_data_int: images,
                                                        all_real_labels: _labels})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        if (iteration < 500) or (iteration % 1000 == 999):
            lib.plot.flush()

            if not os.path.exists(CHECKPOINT_DIR):
                os.mkdir(CHECKPOINT_DIR)
            saver.save(session, os.path.join(CHECKPOINT_DIR, 'model.ckpt'), global_step=iteration)

        lib.plot.tick()

    summary_writer.flush()
    summary_writer.close()
