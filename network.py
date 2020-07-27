import numpy as np
import tensorflow as tf
import cv2 


def network_big(input_size, train=True, android=False):    
    ###########
    # Encoder #
    ###########
    ## 1st line
    if android:
        inputs = tf.keras.layers.Input(input_size)
        # for android 
        inputs_s = tf.keras.layers.Lambda(lambda x: x[:, :, :, :3])(inputs)
        conv1 = tf.keras.layers.Conv2D(8, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs_s)
    else:
        inputs = tf.keras.layers.Input(input_size)
        conv1 = tf.keras.layers.Conv2D(8, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs)
    
    conv1 = residual_block(conv1, filters=8, kernel_size=(3, 3))
    conv1 = tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name="Additional_conv")(conv1)
    conv1 = residual_block_withName(conv1, filters=8, kernel_size=(3, 3), name="residual_block_withName")

    ## 2nd line
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = residual_block(conv2, filters=32, kernel_size=(3, 3))

    ## 3rd line
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = residual_block(conv3, filters=64, kernel_size=(3, 3))

    ## 4th line
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = residual_block(conv4, filters=128, kernel_size=(3, 3))
    conv4 = residual_block(conv4, filters=128, kernel_size=(3, 3))

    ## 5th line
    conv5 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4)
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
    conv4_inv = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
    conv4_inv = tf.keras.layers.Add()([conv4, conv4_inv])
    conv4_inv = residual_block(conv4_inv, filters=128, kernel_size=(3, 3))
    conv4_inv = residual_block(conv4_inv, filters=128, kernel_size=(3, 3))
    conv4_inv = residual_block(conv4_inv, filters=128, kernel_size=(3, 3))

    ## 3rd-inverse line
    conv3_inv = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4_inv)
    conv3_inv = tf.keras.layers.Add()([conv3, conv3_inv])
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3))
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3))
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3))

    ## 2nd-inverse line
    conv2_inv = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3_inv)
    conv2_inv = tf.keras.layers.Add()([conv2, conv2_inv])
    conv2_inv = residual_block(conv2_inv, filters=32, kernel_size=(3, 3))
    conv2_inv = residual_block(conv2_inv, filters=32, kernel_size=(3, 3))
    conv2_inv = residual_block(conv2_inv, filters=32, kernel_size=(3, 3))

    ## 1st-inverse line
    conv1_inv = tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2_inv)
    conv1_inv = tf.keras.layers.Add()([conv1, conv1_inv])
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3))
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3))
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3))

    ### Boundary attention map 추가
    b = tf.keras.layers.Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal', name="boundary_conv1d_")(conv1_inv)
    ba = tf.keras.layers.Activation("sigmoid", name='boundary_attention')(b)
    
    conv6 = tf.keras.layers.Conv2D(3, (1, 1))(conv1_inv)
    x = tf.keras.layers.Activation('tanh')(conv6)
    
    # Concatenating
    x = tf.keras.layers.Concatenate(axis=-1)([x, ba])

    shortcut = x

    x =tf.keras.layers.SeparableConv2D(3, (3, 3), padding='same', depthwise_initializer='he_normal')(x)
    
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        
    x = tf.keras.layers.SeparableConv2D(4, (3, 3), padding='same', depthwise_initializer='he_normal')(x)
    
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Conv2D(1, (1, 1))(x)
    

    if train:
        out = tf.keras.layers.Activation('sigmoid', name='output')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=[out, ba])

    else:
        out = tf.keras.layers.Activation('sigmoid', name='output')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=out)

    return model

def residual_block(x, filters, kernel_size=(3, 3)):
    
    shortcut = x
    
    G_IDX = np.random.randn(1)[0]
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
    
    G_IDX = np.random.randn(1)[0]
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    return x


def residual_block_withName(x, filters, kernel_size=(3, 3), name=""):
    
    shortcut = x

    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name+"_prelu_1")(x)
        
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal', name=name+"_sep_conv_1")(x)

    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name+"_prelu_2")(x)
        
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal', name=name+"_sep_conv_2")(x)

    x = tf.keras.layers.Add(name=name+"_add_1")([shortcut, x])
    return x


def network_small(input_size, train=True, android=False):

    if android:
        inputs = tf.keras.layers.Input(input_size)
        inputs_s = tf.keras.layers.Lambda(lambda x: x[:, :, :, :3])(inputs)
        conv1 = tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs_s)
    else:
        inputs = tf.keras.layers.Input(input_size)
        conv1 = tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs)
    
    conv1 = residual_block(conv1, filters=8, kernel_size=(3, 3))

    conv2 = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = residual_block(conv2, filters=16, kernel_size=(3, 3))

    conv3 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = residual_block(conv3, filters=32, kernel_size=(3, 3))

    conv4 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = residual_block(conv4, filters=64, kernel_size=(3, 3))

    conv5 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4)
    conv5 = residual_block(conv5, filters=128, kernel_size=(3, 3))
    

    conv4_inv = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
    conv4_inv = tf.keras.layers.Add()([conv4, conv4_inv])
    conv4_inv = residual_block(conv4_inv, filters=64, kernel_size=(3, 3))

    conv3_inv = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4_inv)
    conv3_inv = tf.keras.layers.Add()([conv3, conv3_inv])
    conv3_inv = residual_block(conv3_inv, filters=32, kernel_size=(3, 3))
    
    conv2_inv = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3_inv)
    conv2_inv = tf.keras.layers.Add()([conv2, conv2_inv])
    conv2_inv = residual_block(conv2_inv, filters=16, kernel_size=(3, 3))

    conv1_inv = tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2_inv)
    conv1_inv = tf.keras.layers.Add()([conv1, conv1_inv])
    conv1_inv = residual_block(conv1_inv, filters=8, kernel_size=(3, 3))
    
    ### Boundary attention map
    b = tf.keras.layers.Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal', name="boundary_conv1d_")(conv1_inv)
    bf = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(b)
    ba = tf.keras.layers.Activation("sigmoid", name='boundary_attention')(bf)

    fea = tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1_inv)

    if android:
        x = tf.keras.layers.Concatenate(axis=-1)([fea, inputs_s, ba])
    else: 
        x = tf.keras.layers.Concatenate(axis=-1)([fea, inputs, ba])

    shortcut = x

    x = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', depthwise_initializer='he_normal')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.SeparableConv2D(12, (3, 3), padding='same', depthwise_initializer='he_normal')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    x = tf.keras.layers.Add()([shortcut, x])

    x = tf.keras.layers.Conv2D(1, (1, 1))(x)

    out = tf.keras.layers.Activation('sigmoid', name='output')(x)
    
    if train:
        model = tf.keras.models.Model(inputs=inputs, outputs=[out, ba])

    else :
        model = tf.keras.models.Model(inputs=inputs, outputs=out)

    return model