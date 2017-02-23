# -*- coding: utf-8 -*-

"""
   Inception V3 model
"""

from operations import *

from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.core import Dropout, Lambda
from keras.engine.topology import merge
import warnings
from keras.layers.pooling import GlobalAveragePooling2D


def InceptionV3(include_top=True, weights='imagenet',
                input_tensor=None, input_shape=None,
                weight_decay=0.00004, num_classes=1000,
                dropout_prob=0., aux_include=True):
    """Inception v3 architecture
       Note that the default image size for this model is 299x299
    """

    if input_shape is None:
        input_shape = (299, 299)

    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + input_shape
        channel_axis = 1
    else:
        input_shape = input_shape + (3,)
        channel_axis = 3

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    # Using `tf` order
    # 299 x 299 x 3
    x = conv2d_bn(img_input, 32, 3, 3, subsample=(2, 2),
                  border_mode='valid', weight_decay=weight_decay,
                  name='0')

    # 149 x 149 x 32
    x = conv2d_bn(x, 32, 3, 3, border_mode='valid',
                  weight_decay=weight_decay, name='1')

    # 147 x 147 x 32
    x = conv2d_bn(x, 64, 3, 3, weight_decay=weight_decay, name='2')

    # 147 x 147 x 64
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_1')(x)

    # 73  x 73 x 64
    x = conv2d_bn(x, 80, 1, 1, weight_decay=weight_decay, name='3')

    # 73 x 73 x 80
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid',
                  weight_decay=weight_decay, name='4')

    # 71 x 71 x 192
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_2')(x)

    # 35 x 35 x 192
    # Inception block
    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, weight_decay=weight_decay)

    branch5x5 = conv2d_bn(x, 48, 1, 1, weight_decay=weight_decay)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, weight_decay=weight_decay)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, weight_decay=weight_decay)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, weight_decay=weight_decay)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, weight_decay=weight_decay)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, weight_decay=weight_decay)

    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed_0')

    for i in range(2):
        branch1x1 = conv2d_bn(x, 64, 1, 1, weight_decay=weight_decay)

        branch5x5 = conv2d_bn(x, 48, 1, 1, weight_decay=weight_decay)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, weight_decay=weight_decay)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1, weight_decay=weight_decay)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 96, 3, 3, weight_decay=weight_decay)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 96, 3, 3, weight_decay=weight_decay)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(
            branch_pool, 64, 1, 1, weight_decay=weight_decay)

        x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed_' + str(i + 1))

    # mixed_3: 17 x 17 x 768
    branch3x3 = conv2d_bn(
        x, 384, 3, 3, subsample=(2, 2),
        border_mode='valid', weight_decay=weight_decay)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, weight_decay=weight_decay)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, weight_decay=weight_decay)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, subsample=(2, 2),
        border_mode='valid', weight_decay=weight_decay)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
    x = merge([branch3x3, branch3x3dbl, branch_pool], mode='concat',
              concat_axis=channel_axis, name='mixed_3')

    # mixed_4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, weight_decay=weight_decay)

    branch7x7 = conv2d_bn(x, 128, 1, 1, weight_decay=weight_decay)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, weight_decay=weight_decay)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, weight_decay=weight_decay)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, weight_decay=weight_decay)
    branch7x7dbl = conv2d_bn(
        branch7x7dbl, 128, 7, 1, weight_decay=weight_decay)
    branch7x7dbl = conv2d_bn(
        branch7x7dbl, 128, 1, 7, weight_decay=weight_decay)
    branch7x7dbl = conv2d_bn(
        branch7x7dbl, 128, 7, 1, weight_decay=weight_decay)
    branch7x7dbl = conv2d_bn(
        branch7x7dbl, 192, 1, 7, weight_decay=weight_decay)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, weight_decay=weight_decay)

    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis, name='mixed_4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1, weight_decay=weight_decay)

        branch7x7 = conv2d_bn(x, 160, 1, 1, weight_decay=weight_decay)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7, weight_decay=weight_decay)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, weight_decay=weight_decay)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, weight_decay=weight_decay)
        branch7x7dbl = conv2d_bn(
            branch7x7dbl, 160, 7, 1, weight_decay=weight_decay)
        branch7x7dbl = conv2d_bn(
            branch7x7dbl, 160, 1, 7, weight_decay=weight_decay)
        branch7x7dbl = conv2d_bn(
            branch7x7dbl, 160, 7, 1, weight_decay=weight_decay)
        branch7x7dbl = conv2d_bn(
            branch7x7dbl, 192, 1, 7, weight_decay=weight_decay)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(
            branch_pool, 192, 1, 1, weight_decay=weight_decay)

        x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed_' + str(i + 5))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, weight_decay=weight_decay)

    branch7x7 = conv2d_bn(x, 192, 1, 1, weight_decay=weight_decay)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7, weight_decay=weight_decay)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, weight_decay=weight_decay)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, weight_decay=weight_decay)
    branch7x7dbl = conv2d_bn(
        branch7x7dbl, 192, 7, 1, weight_decay=weight_decay)
    branch7x7dbl = conv2d_bn(
        branch7x7dbl, 192, 1, 7, weight_decay=weight_decay)
    branch7x7dbl = conv2d_bn(
        branch7x7dbl, 192, 7, 1, weight_decay=weight_decay)
    branch7x7dbl = conv2d_bn(
        branch7x7dbl, 192, 1, 7, weight_decay=weight_decay)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, weight_decay=weight_decay)

    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed_7')

    if aux_include:
        # Auxiliary Head logits
        aux_classifier = AveragePooling2D(
            (5, 5), strides=(3, 3), border_mode='valid')(x)
        aux_classifier = conv2d_bn(
            aux_classifier, 128, 1, 1, weight_decay=weight_decay)

        # Shape of feature map before the final layer
        # shape = aux_classifier.output_shape
        aux_classifier = conv2d_bn(aux_classifier, 768, 5, 5,
                                   border_mode='valid',
                                   weight_decay=weight_decay)

        aux_classifier = Flatten()(aux_classifier)

        if weight_decay and weight_decay > 0:
            aux_classifier = Dense(num_classes, activation='softmax',
                                   W_regularizer=l2(weight_decay),
                                   name='aux_classifier')(aux_classifier)
        else:
            aux_classifier = Dense(
                num_classes, activation='softmax',
                name='aux_classifier')(aux_classifier)

    # mixed 8: 8 x 8 x 1280.
    branch3x3 = conv2d_bn(x, 192, 1, 1, weight_decay=weight_decay)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          subsample=(2, 2),
                          border_mode='valid',
                          weight_decay=weight_decay)

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, weight_decay=weight_decay)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7, weight_decay=weight_decay)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1, weight_decay=weight_decay)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            subsample=(2, 2),
                            border_mode='valid',
                            weight_decay=weight_decay)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)

    x = merge([branch3x3, branch7x7x3, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed_8')

    # mixed 9 10: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1, weight_decay=weight_decay)

        branch3x3 = conv2d_bn(x, 384, 1, 1, weight_decay=weight_decay)
        branch3x3 = merge([conv2d_bn(branch3x3, 384, 1, 3,
                                     weight_decay=weight_decay),
                           conv2d_bn(branch3x3, 384, 3, 1,
                                     weight_decay=weight_decay)],
                          mode='concat', concat_axis=channel_axis)

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, weight_decay=weight_decay)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 384, 3, 3, weight_decay=weight_decay)
        branch3x3dbl = merge([conv2d_bn(branch3x3dbl, 384, 1, 3,
                                        weight_decay=weight_decay),
                              conv2d_bn(branch3x3dbl, 384, 3, 1,
                                        weight_decay=weight_decay)],
                             mode='concat', concat_axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(
            branch_pool, 192, 1, 1, weight_decay=weight_decay)

        x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed_' + str(9 + i))

    # Dimension reduction
    # 2048 x 8 x 8
    x = conv2d_bn(x, 1024, 1, 1,
                  weight_decay=weight_decay)

    # Final pooling and prediction
    # 1024 x 8 x 8
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_prob)(x)

    # 1024
    if weight_decay and weight_decay > 0:
        predictions = Dense(num_classes,
                            activation='softmax',
                            W_regularizer=l2(weight_decay),
                            name='predictions')(x)
    else:
        predictions = Dense(num_classes,
                            activation='softmax',
                            name='predictions')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if aux_include:
        model = Model(
            inputs, [predictions, aux_classifier],
            name='inception_v3_with_aux')
    else:
        model = Model(inputs, predictions, name='inception_v3')

    return model
