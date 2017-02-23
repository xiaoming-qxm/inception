# -*- coding: utf-8 -*-

"""
   Usage examples for Inception v3 on CIFAR-10 dataset
"""

from inception_v3 import *
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import cv2
import numpy as np


def load_data():
    (_, _), (x_train, y_train) = cifar10.load_data()
    x_train = x_train[:100]
    y_train = y_train[:100]
    print(x_train.shape)

    data_upscaled = np.zeros((100, 3, 299, 299))

    for i, img in enumerate(x_train):
        im = img.transpose((1, 2, 0))
        large_img = cv2.resize(
            im, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img.transpose((2, 0, 1))

    y_train = to_categorical(y_train, 10)

    return data_upscaled, y_train


def main():

    x_train, y_train = load_data()
    print(x_train.shape)

    model = InceptionV3(num_classes=10)

    # model.summary()
    model.compile(optimizer='rmsprop',
                  loss={'predictions': 'categorical_crossentropy',
                        'aux_classifier': 'categorical_crossentropy'},
                  loss_weights={'predictions': 1., 'aux_classifier': 0.2})

    model.fit(x_train, {'predictions': y_train, 'aux_classifier': y_train},
              nb_epoch=50, batch_size=8)

main()
