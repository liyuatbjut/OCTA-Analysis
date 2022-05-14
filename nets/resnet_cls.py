from __future__ import print_function

import keras

from keras import layers
from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.layers import Lambda, Reshape, Multiply, Concatenate
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

from PIL import Image
from utils.utils import get_heatmap_mask
import numpy as np
import tensorflow as tf


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):

    img_input = Input(shape=input_shape)

    x_inp = ZeroPadding2D((3, 3))(img_input)
    x_inp = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x_inp)
    x_inp = BatchNormalization(name='bn_conv1')(x_inp)
    x_inp = Activation('relu')(x_inp)
    x_inp = MaxPooling2D((3, 3), strides=(2, 2))(x_inp)

    # channels split
    img_mask = Lambda(lambda x: x[:, :, :, 0])(img_input)
    img_origin = Lambda(lambda x: x[:, :, :, 1])(img_input)
    img_mask_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(img_mask)
    img_origin_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(img_origin)

    img_mask_to_conv = ZeroPadding2D((1, 1))(img_mask_exp)
    img_mask_to_conv = Conv2D(32, (3, 3), strides=(2, 2), name='conv1_a')(img_mask_to_conv)
    img_mask_to_conv = BatchNormalization(name='bn_xmask')(img_mask_to_conv)
    img_mask_to_conv = Activation('relu')(img_mask_to_conv)

    img_origin_to_conv = ZeroPadding2D((1, 1))(img_origin_exp)
    img_origin_to_conv = Conv2D(32, (3, 3), strides=(2, 2), name='conv1_b')(img_origin_to_conv)
    img_origin_to_conv = BatchNormalization(name='bn_xoriginal')(img_origin_to_conv)
    img_origin_to_conv = Activation('relu')(img_origin_to_conv)

    # fusion
    x = Concatenate(axis=-1)([img_mask_to_conv, img_origin_to_conv])

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), name='conv_new')(x)
    x = BatchNormalization(name='bn_new')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Concatenate(axis=-1)([x, x_inp])

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    model = Model(img_input, x, name='resnet50')

    return model


if __name__ == '__main__':
    model = ResNet50()



