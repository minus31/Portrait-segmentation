import numpy as np
from keras.layers import *
from keras.models import *
from keras.layers import Conv2D, SeparableConv2D, Conv2DTranspose, add, ReLU, Dropout, Reshape, Permute, Input, Activation, Add, BatchNormalization
from keras.layers import Lambda, Concatenate
from keras.models import Model
from keras.activations import sigmoid
from keras import backend as K
import tensorflow as tf


def matting_net(input_size, batchnorm=False, android=False):
    ###########
    # Encoder #
    ###########
    ## 1st line
    if android:
        inputs = Input(input_size)
        # for android 
        inputs_s = Lambda(lambda x: x[:, :, :, :3])(inputs)
        conv1 = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(inputs_s)
    else:
        inputs = Input(input_size)
        conv1 = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    
    conv1 = residual_block(conv1, filters=8, kernel_size=(3, 3), batchnorm=batchnorm)

    ## 2nd line
    conv2 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = residual_block(conv2, filters=32, kernel_size=(3, 3), batchnorm=batchnorm)

    ## 3rd line
    conv3 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = residual_block(conv3, filters=64, kernel_size=(3, 3), batchnorm=batchnorm)

    ## 4th line
    conv4 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = residual_block(conv4, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)
    conv4 = residual_block(conv4, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)

    ## 5th line
    conv5 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4)
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)

    ###########
    # Decoder #
    ###########
    ## 4th-inverse line
    conv4_inv = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
    conv4_inv = add([conv4, conv4_inv])
    conv4_inv = residual_block(conv4_inv, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)
    conv4_inv = residual_block(conv4_inv, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)
    conv4_inv = residual_block(conv4_inv, filters=128, kernel_size=(3, 3), batchnorm=batchnorm)

    ## 3rd-inverse line
    conv3_inv = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4_inv)
    conv3_inv = add([conv3, conv3_inv])
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3), batchnorm=batchnorm)
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3), batchnorm=batchnorm)
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3), batchnorm=batchnorm)

    ## 2nd-inverse line
    conv2_inv = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3_inv)
    conv2_inv = add([conv2, conv2_inv])
    conv2_inv = residual_block(conv2_inv, filters=32, kernel_size=(3, 3), batchnorm=batchnorm)
    conv2_inv = residual_block(conv2_inv, filters=32, kernel_size=(3, 3), batchnorm=batchnorm)
    conv2_inv = residual_block(conv2_inv, filters=32, kernel_size=(3, 3), batchnorm=batchnorm)

    ## 1st-inverse line
    conv1_inv = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2_inv)
    conv1_inv = add([conv1, conv1_inv])
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3), batchnorm=batchnorm)
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3), batchnorm=batchnorm)
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3), batchnorm=batchnorm)
    
    ### last sigmoid function
    if input_size[0] == 128:
        conv6 = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1_inv)
        conv6 = Conv2D(3, (1, 1))(conv6)
    else:
        conv6 = Conv2D(3, (1, 1))(conv1_inv)
    
    # Fs = Lambda(lambda x: x[:, :, :, 0:1])(conv6)
    # Us = Lambda(lambda x: x[:, :, :, 1:2])(conv6)
    # Bs = Lambda(lambda x: x[:, :, :, 2:])(conv6)
    
    # Fs = Lambda(lambda x : K.exp(x))(Fs)
    # Us = Lambda(lambda x : K.exp(x))(Us)
    # Bs = Lambda(lambda x : K.exp(x))(Bs)

    # s_exp = Add(name="add_exps")([Fs, Us, Bs])

    # div_Fs = Lambda(lambda x : x[0] / x[1])([Fs, s_exp])
    # div_Us = Lambda(lambda x : x[0] / x[1])([Us, s_exp])
    # div_Bs = Lambda(lambda x : x[0] / x[1])([Bs, s_exp])

    # x = Concatenate(axis=-1)([div_Fs, div_Us, div_Bs])

    x = Activation('tanh')(conv6)
    
    shortcut = x
    x = ReLU(name='re_lu_24')(x)
    x = SeparableConv2D(3, (3, 3), padding='same', depthwise_initializer='he_normal', name='separable_conv2d_47')(x)
    x = Activation('relu', name='activation_27')(x)
    x = SeparableConv2D(3, (3, 3), padding='same', depthwise_initializer='he_normal', name='separable_conv2d_48')(x)
    x = Add(name='add_28')([shortcut, x])

    x = Conv2D(1, (1, 1), name='conv2d_7')(x)

    out = Activation('sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=out)
    return model



def residual_block(x, filters, kernel_size=(3, 3), batchnorm=False):
    shortcut = x
    x = ReLU()(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
    x = add([shortcut, x])
    return x
