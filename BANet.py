import tensorflow as tf 
import numpy as np 


def residual_block(x, filters=64, kernel_size=(3, 3)):
    shortcut = x
    x = tf.keras.layers.SeparableConv2D(filters / 2, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer='he_normal')(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Add()([shortcut, x])
    return x

def BAnet(input_size, train=True, android=False):
    ###########
    # Encoder #
    ###########
    ## 1st line
    
    # input shape -> 1/2
    if android:
        inputs = tf.keras.layers.Input(shape=input_size)
        # for android 
        inputs_s = tf.keras.layers.Lambda(lambda x: x[:, :, :, :3])(inputs)
        conv1 = tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs_s)
    else:
        inputs = tf.keras.layers.Input(shape=input_size)
        conv1 = tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs)
    
    # 1/2 -> 1/4
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = residual_block(conv1, filters=64, kernel_size=(3, 3))

    ## 2nd line 1/4 -> 1/8
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = residual_block(conv2, filters=64, kernel_size=(3, 3))

    ## 3rd line 1/8 -> 1/16
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = residual_block(conv3, filters=64, kernel_size=(3, 3))

    ## 4th line 1/16 -> 1/32
    conv4 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = residual_block(conv4, filters=64, kernel_size=(3, 3))

    ###########
    # Decoder #
    ###########
    
    # 1/32 -> 1/16
    ## 4th-inverse line
    conv3_inv = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4)
    conv3_inv = tf.keras.layers.Add()([conv3, conv3_inv])
    conv3_inv = residual_block(conv3_inv, filters=64, kernel_size=(3, 3))

    # 1/16 -> 1/8
    ## 3rd-inverse line
    conv2_inv = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv4_inv)
    conv2_inv = tf.keras.layers.Add()([conv2, conv2_inv])
    conv2_inv = residual_block(conv2_inv, filters=64, kernel_size=(3, 3))

    # 1/8 -> 1/4
    ## 2nd-inverse line
    conv1_inv = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv3_inv)
    conv1_inv = tf.keras.layers.Add()([conv1, conv1_inv])
    conv1_inv = residual_block(conv1_inv, filters=64, kernel_size=(3, 3))
    
    
    ### for boundary mapping 
    # prejection to 1 channel
    ba_projection = tf.keras.layers.Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(conv1_inv)
    # Upsample to input size
    ba_comparison = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(ba_projection)
    
    ba_map = tf.keras.layers.Activation("sigmoid")(ba_comparison)
    
    ### Feature Fusion Module
    semantic = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv1_inv)
    semantic = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(semantic)
    
    if android:
        x = tf.keras.layers.Concatenate(axis=-1)([semantic, inputs_s, ba_map])
    else: 
        x = tf.keras.layers.Concatenate(axis=-1)([semantic, inputs, ba_map])
        
    tf.keras.layers.Conv2D(filters=)

    shortcut = x
    
    x = tf.keras.layers.Activation('relu')(x)
    x = SeparableConv2D(3, (3, 3), padding='same', depthwise_initializer='he_normal', name='separable_conv2d_47_')(x)
    x = Activation('relu', name='activation_27')(x)
    x = SeparableConv2D(4, (3, 3), padding='same', depthwise_initializer='he_normal', name='separable_conv2d_48_')(x)
    x = Add(name='add_28')([shortcut, x])
    x = Conv2D(1, (1, 1), name='conv2d_7_')(x)
    out = Activation('sigmoid', name='output')(x)
    
    if train:
        model = Model(inputs=inputs, outputs=[out, ba])

    else :
        model = Model(inputs=inputs, outputs=out)

    return model