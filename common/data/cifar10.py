import numpy as np

import os
import urllib
import gzip
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        # print('-------------dict.keys()---------------')
        # print(dict.keys())
        # print('-------------dict.keys()---------------')
    return dict[b'data'], dict[b'labels']


def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(os.path.join(data_dir, filename))
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'], batch_size, data_dir),
        cifar_generator(['test_batch'], batch_size, data_dir)
    )
