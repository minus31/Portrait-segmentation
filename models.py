import numpy as np
from keras.layers import *
from keras.models import *
from keras.layers import Conv2D, SeparableConv2D, Conv2DTranspose, add, ReLU, Dropout, Reshape, Permute, Input, Activation, Add, BatchNormalization
from keras.layers import Lambda, Concatenate
from keras.models import Model
from keras.activations import sigmoid
from keras import backend as K
import tensorflow as tf
import cv2 


def matting_net(input_size, train=True, android=False):
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
    
    conv1 = residual_block(conv1, filters=8, kernel_size=(3, 3))

    ## 2nd line
    conv2 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = residual_block(conv2, filters=32, kernel_size=(3, 3))

    ## 3rd line
    conv3 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = residual_block(conv3, filters=64, kernel_size=(3, 3))

    ## 4th line
    conv4 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = residual_block(conv4, filters=128, kernel_size=(3, 3))
    conv4 = residual_block(conv4, filters=128, kernel_size=(3, 3))

    ## 5th line
    conv5 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4)
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3))
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3))
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3))
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3))
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3))
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3))

    ###########
    # Decoder #
    ###########
    ## 4th-inverse line
    conv4_inv = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
    conv4_inv = add([conv4, conv4_inv])
    conv4_inv = residual_block(conv4_inv, filters=128, kernel_size=(3, 3))
    conv4_inv = residual_block(conv4_inv, filters=128, kernel_size=(3, 3))
    conv4_inv = residual_block(conv4_inv, filters=128, kernel_size=(3, 3))

    ## 3rd-inverse line
    conv3_inv = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4_inv)
    conv3_inv = add([conv3, conv3_inv])
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3))
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3))
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3))

    ## 2nd-inverse line
    conv2_inv = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3_inv)
    conv2_inv = add([conv2, conv2_inv])
    conv2_inv = residual_block(conv2_inv, filters=32, kernel_size=(3, 3))
    conv2_inv = residual_block(conv2_inv, filters=32, kernel_size=(3, 3))
    conv2_inv = residual_block(conv2_inv, filters=32, kernel_size=(3, 3))

    ## 1st-inverse line
    conv1_inv = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2_inv)
    conv1_inv = add([conv1, conv1_inv])
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3))
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3))
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3))

    ### Boundary attention map 추가
    b = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal', name="boundary_conv1d_")(conv1_inv)
    ba = Activation("sigmoid", name='boundary_attention')(b)
    
    conv6 = Conv2D(3, (1, 1))(conv1_inv)
    x = Activation('tanh')(conv6)
    
    # Concatenating
    x = Concatenate(axis=-1)([x, ba])

    shortcut = x
    
    x = ReLU(name='re_lu_24')(x)
    x = SeparableConv2D(3, (3, 3), padding='same', depthwise_initializer='he_normal', name='separable_conv2d_47_')(x)
    x = Activation('relu', name='activation_27')(x)
    x = SeparableConv2D(4, (3, 3), padding='same', depthwise_initializer='he_normal', name='separable_conv2d_48_')(x)
    x = Add(name='add_28')([shortcut, x])
    x = Conv2D(1, (1, 1), name='conv2d_7_')(x)
    
    # if train:
    #     out = Activation('sigmoid', name='output')(x)
    #     model = Model(inputs=inputs, outputs=[out, ba])
    if train:
        # out = Activation('sigmoid', name='output')(x)
        out = Activation('tanh', name='output')(x)
        out = Activation('relu', name='output')(out)
        # refine = Lambda(lambda x : refine_loss(x[0], x[1], x[2]), name='refine')([out, inputs, ba])
        model = Model(inputs=inputs, outputs=[out, ba])

    else:
        # out = Activation('sigmoid', name='output')(x)
        out = Activation('tanh', name='output')(x)
        out = Activation('relu', name='output')(out)
        model = Model(inputs=inputs, outputs=out)

    return model

def residual_block(x, filters, kernel_size=(3, 3)):
    shortcut = x
    x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
    x = ReLU()(x)
    x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = add([shortcut, x])
    return x

def compute_gredient(src, color=True):
    # if color :
    #     GX = tf.constant(np.array([[[1,1,1], [0,0,0], [-1,-1,-11]],
    #                            [[2, 2, 2], [0,0,0], [-2,-2,-2]],
    #                            [[1,1,1], [0,0,0] ,[-1,-1,-1]]]), tf.float32)

    #     GY = tf.constant(np.array([[[1,1,1], [2, 2, 2], [1,1,1]],
    #                             [[0,0,0], [0,0,0], [0,0,0]],
    #                             [[-1,-1,-1], [-2,-2,-2],[-1,-1,-1]]]), tf.float32)

    #     GX = tf.reshape(GX, (3,3,3,1))
    #     GY = tf.reshape(GY, (3,3,3,1))

    # else : 
    #     GX = tf.constant(np.array([[1, 0, -1],
    #                             [2, 0, -2],
    #                             [1, 0 ,-1]]), tf.float32)

    #     GY = tf.constant(np.array([[1, 2, 1],
    #                             [0, 0, 0],
    #                             [-1, -2,-1]]), tf.float32)
    
    #     GX = tf.reshape(GX, (3,3,1,1))
    #     GY = tf.reshape(GY, (3,3,1,1))
    GX = tf.constant(np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0 ,-1]]), tf.float32)

    GY = tf.constant(np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2,-1]]), tf.float32)

    GX = tf.reshape(GX, (3,3,1,1))
    GY = tf.reshape(GY, (3,3,1,1))

    if src.shape[-1] > 1:
        src = tf.reduce_mean(src, axis=-1)
        src = tf.expand_dims(src, axis=-1)

    X_g = tf.nn.conv2d(src, GX, padding="SAME")
    Y_g = tf.nn.conv2d(src, GY, padding="SAME")

    M = tf.sqrt(tf.add(tf.pow(X_g, 2), tf.pow(Y_g, 2)))

    nu_x = X_g / M 
    nu_y = Y_g / M
    return M, nu_x, nu_y

def refine_loss(y_pred, input_, boundary):

    M_img, nuX_img, nuY_img = compute_gredient(input_, color=True)

    M_pred, nuX_pred, nuY_pred = compute_gredient(y_pred, color=False)

    Lcos = tf.add(1., -1 * tf.abs(tf.add(tf.multiply(nuX_img, nuX_pred), tf.multiply(nuY_img, nuY_pred)))) * M_pred

    Lmag = tf.maximum(tf.add(1.5 * M_img, -1. * M_pred), 0)

    L_refine = tf.multiply(tf.add(tf.multiply(Lcos, 0.5), tf.multiply(Lmag, 0.5)), boundary)
    # res = tf.reduce_mean(L_refine[L_refine > 0], axis=-1)
    # res = tf.reshape(res, (-1, 1))
    return L_refine

# def residual_block(x, filters, kernel_size=(3, 3)):
#     shortcut = x
#     x = ReLU()(x)
#     x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
#     x = add([shortcut, x])
#     return x


def light_matting_net(input_size, train=True, android=False):

    if android:
        inputs = Input(input_size)
        inputs_s = Lambda(lambda x: x[:, :, :, :3])(inputs)
        conv1 = Conv2D(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs_s)
    else:
        inputs = Input(input_size)
        conv1 = Conv2D(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs)
    
    conv1 = residual_block(conv1, filters=8, kernel_size=(3, 3))

    conv2 = Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = residual_block(conv2, filters=16, kernel_size=(3, 3))

    conv3 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = residual_block(conv3, filters=32, kernel_size=(3, 3))

    conv4 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = residual_block(conv4, filters=64, kernel_size=(3, 3))

    conv5 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4)
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3))
    

    conv4_inv = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
    conv4_inv = add([conv4, conv4_inv])
    conv4_inv = residual_block(conv4_inv, filters=64, kernel_size=(3, 3))

    conv3_inv = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4_inv)
    conv3_inv = add([conv3, conv3_inv])
    conv3_inv = residual_block(conv3_inv, filters=32, kernel_size=(3, 3))
    
    conv2_inv = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3_inv)
    conv2_inv = add([conv2, conv2_inv])
    conv2_inv = residual_block(conv2_inv, filters=16, kernel_size=(3, 3))

    conv1_inv = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2_inv)
    conv1_inv = add([conv1, conv1_inv])
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3))
    
    ### Boundary attention map
    b = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal', name="boundary_conv1d_")(conv1_inv)
    bf = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(b)
    ba = Activation("sigmoid", name='boundary_attention')(bf)

    fea = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1_inv)

    if android:
        x = Concatenate(axis=-1)([fea, inputs_s, ba])
    else: 
        x = Concatenate(axis=-1)([fea, inputs, ba])

    shortcut = x

    x = SeparableConv2D(16, (3, 3), padding='same', depthwise_initializer='he_normal', name='separable_conv2d_47_')(x)
    x = ReLU(name='re_lu_24')(x)
    x = SeparableConv2D(12, (3, 3), padding='same', depthwise_initializer='he_normal', name='separable_conv2d_48_')(x)
    x = Activation('relu', name='activation_27')(x)

    x = Add(name='add_28')([shortcut, x])

    x = Conv2D(1, (1, 1), name='conv2d_7_')(x)

    out = Activation('sigmoid', name='output')(x)
    
    if train:
        model = Model(inputs=inputs, outputs=[out, ba])

    else :
        model = Model(inputs=inputs, outputs=out)

    return model

