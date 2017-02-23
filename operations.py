# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Convolution2D, BatchNormalization, Dense
from keras.layers.core import Activation
from keras.regularizers import l2


def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              batch_norm=True, activation='relu',
              weight_decay=0, name=None):
    """Utility function to apply conv + BN(optionally)
    """
    if name is not None:
        bn_name = 'bn_' + name
        conv_name = 'conv_' + name
    else:
        bn_name = None
        conv_name = None
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3

    if weight_decay and weight_decay > 0:
        x = Convolution2D(nb_filter, nb_row, nb_col,
                          subsample=subsample,
                          activation='relu',
                          W_regularizer=l2(weight_decay),
                          border_mode=border_mode,
                          name=conv_name)(x)
    else:
        x = Convolution2D(nb_filter, nb_row, nb_col,
                          subsample=subsample,
                          activation='relu',
                          border_mode=border_mode,
                          name=conv_name)(x)

    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name=bn_name)(x)

    if activation:
        x = Activation(activation)(x)

    return x
