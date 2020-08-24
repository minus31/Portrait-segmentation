"""
REFERENCE : https://github.com/Tramac/Fast-SCNN-pytorch/blob/master/utils/loss.py

I will use input_shape with "224, 168"

FastSCNN 
 - learning to downsample 
 -> global featrue extractor
 - feature_fusion (learning to downsample, global featrue extractor)
 - Classifier (feature_fusion)
 - Interpolation
"""

"""
MODULE LIST 

- ConvBNReLU          : conv2d BN Relu
- DSConv              : depthwise + BN + ReLU + pointwise + BN ReLU
- DWConv              : Depthwise + BN + ReLU 
- Linear BottleNeck   : x -> CONVBNRELU  + DWConv + Pointwise + BN -> y + x
- PyramidPooling      : pool = adaptiveAvgPool2D, conv=CONVBNRELU, upsampling=Interpolation
"""

import tensorflow as tf 
import numpy as np
from tensorflow.keras.mixed_precision import experimental as mixed_precision

def swish_activation(x):
    sig = tf.keras.layers.Activation("sigmoid")(x)
    out = tf.keras.layers.Multiply()([x, sig])
    return out

def _ConvBNReLU(x, out_ch, k_size=3, stride=1, padding="same", **kwargs):
    # add regularization layers and initializer
    x = tf.keras.layers.Conv2D(out_ch, 
                               k_size, 
                               stride, 
                               padding, 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x 

def _DWConv(x, k_size=3, stride=1, padding="same", **kwargs):
    x = tf.keras.layers.DepthwiseConv2D(k_size, stride, padding,
                                        kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def _DSConv(x, out_ch, stride=1):
    x = _DWConv(x, stride=stride)
    x = tf.keras.layers.Conv2D(out_ch, 
                               kernel_size=1,
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def linearBottlenect(x, out_ch, t=6, stride=1, **kwargs):
    use_shortcut = (stride == 1 and x.shape[-1] == out_ch)
    # point-wise
    out = _ConvBNReLU(x, x.shape[-1]*t, 1)
    # depth-wise
    out = _DWConv(out, k_size=3, stride=stride)
    out = tf.keras.layers.Conv2D(out_ch, 
                               kernel_size=1, 
                               use_bias=False,
                               kernel_initializer='he_normal')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    if use_shortcut:
        out = x + out
    return out

class BilinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, size):
        super(BilinearInterpolation, self).__init__()
        self.size = size
        
    def call(self, x):
        self.out = tf.compat.v1.image.resize_bilinear(x, self.size, align_corners=True)
        return self.out

def __pyramid_module(x, pool_size, inter_ch, **kwargs):
    size = x.shape[-3:-1]
    x = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)
    x = _ConvBNReLU(x, inter_ch, k_size=1)
    x = BilinearInterpolation(size)(x)
    return x 

def pyramidPooling(x, out_ch, **kwargs):
    inter_ch = x.shape[-1] // 4
    fea1 = __pyramid_module(x, pool_size=1, inter_ch=inter_ch)
    fea2 = __pyramid_module(x, pool_size=2, inter_ch=inter_ch)
    fea3 = __pyramid_module(x, pool_size=3, inter_ch=inter_ch)
    fea6 = __pyramid_module(x, pool_size=4, inter_ch=inter_ch)
    x = tf.keras.layers.Concatenate(axis=-1)([x, fea1, fea2, fea3, fea6])
    x = _ConvBNReLU(x, out_ch, k_size=1)
    return x
    
def learningToDownsample(x, dw_ch1=32, dw_ch2=48, out_ch=64, **kwargs):
    x = _ConvBNReLU(x, dw_ch1, k_size=3, stride=2)
    x = _DSConv(x, dw_ch2, stride=2)
    x = _DSConv(x, out_ch, stride=2)
    return x

def _block_layer(x, block, out_ch, num_block, t=6, stride=1):
    x = block(x, out_ch, t=6, stride=stride)
    for i in range(1, num_block):
        x = block(x, out_ch, t=6, stride=1)
    return  x
    
def globalFeatureExtractor(x, block_channels=[64, 96, 128], out_ch=128, t=6, num_block=(3, 3, 3), **kwargs):
    x = _block_layer(x, linearBottlenect, block_channels[0], num_block[0], t, stride=2)
    x = _block_layer(x, linearBottlenect, block_channels[1], num_block[1], t, stride=2)
    x = _block_layer(x, linearBottlenect, block_channels[2], num_block[2], t, stride=1)
    x = pyramidPooling(x, out_ch)
    return x 

def featureFusionModule(high, low, out_ch, scale_factor=4, **kwargs):
    size = np.array(high.shape[-3:-1])
    low = BilinearInterpolation(size)(low)
    low = _DWConv(low, k_size=3, stride=1)
    low = tf.keras.layers.Conv2D(out_ch,
                               kernel_size=1,
                               kernel_initializer='he_normal')(low)
    low = tf.keras.layers.BatchNormalization()(low)
    
    high = tf.keras.layers.Conv2D(out_ch,
                               kernel_size=1,
                               kernel_initializer='he_normal')(high)
    high = tf.keras.layers.BatchNormalization()(high)
    
    out = high + low 
    out = tf.keras.layers.Activation("relu")(out)
    return out
    
def classifier(x, num_classes, stride=1, train=True, **kwargs):
    out_ch = x.shape[-1]
    x = _DSConv(x, out_ch, stride)
    x = _DSConv(x, out_ch, stride)
    # for boundary 
    b = tf.keras.layers.Conv2D(num_classes,
                               kernel_size=1, 
                               kernel_initializer='he_normal')(x)
    b = tf.keras.layers.Activation("sigmoid", dtype='float32')(b)
    
    x = _DSConv(x, 3, stride)
    x = tf.keras.layers.Concatenate()([x, b])
    x = _DSConv(x, out_ch//4, stride)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(num_classes,
                               kernel_size=1, 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation("sigmoid", dtype='float32')(x)
    if train :
        return x, b
    return x

def fastSCNN(input_shape=(256, 192, 3), train=True):
    input_ = tf.keras.layers.Input(shape=input_shape)
    down = learningToDownsample(input_, dw_ch1=32, dw_ch2=48, out_ch=64)
    gf = globalFeatureExtractor(down)
    fus = featureFusionModule(down, gf, out_ch=128)
    cls = classifier(fus, num_classes=1, train=train)

    if train:
        output_c, output_b = cls
        # output_c = BilinearInterpolation(input_shape[:2])(cls)
        # output_b = BilinearInterpolation(input_shape[:2])(boundary)
        for _ in range(3):
            output_c = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same")(output_c)
            output_b = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same")(output_b)

        output_c = tf.keras.layers.Activation("sigmoid")(output_c)
        output_b = tf.keras.layers.Activation("sigmoid")(output_b)
        output = [output_c, output_b]
    else: 
        # output = BilinearInterpolation(input_shape[:2])(cls)
        output = cls
        for _ in range(3):
            output = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same")(output)
        output = tf.keras.layers.Activation("sigmoid")(output)
    return tf.keras.models.Model(input_, output)
    

###################################################################

def classifier_ori(x, num_classes, stride=1, train=True, **kwargs):
    out_ch = x.shape[-1]
    x = _DSConv(x, out_ch, stride)
    x = _DSConv(x, out_ch, stride)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(num_classes,
                               kernel_size=1, 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation("sigmoid", dtype='float32')(x)
    return x


def fastSCNN_edition2(input_shape=(256, 192, 3), train=True):
    input_ = tf.keras.layers.Input(shape=input_shape)
    down = learningToDownsample(input_, dw_ch1=32, dw_ch2=48, out_ch=64)
    gf = globalFeatureExtractor(down)
    fus = featureFusionModule(down, gf, out_ch=128)
    b = classifier_ori(fus, num_classes=1)
    c = classifier_ori(fus, num_classes=1)
    # input_low = BilinearInterpolation([input_shape[0]//8, input_shape[1]//8])(input_)
    b_c = tf.keras.layers.Concatenate(axis=-1)([b, c])
    # c_out = tf.keras.layers.Conv2D(1,
    #                            kernel_size=3,
    #                            padding='same',
    #                            kernel_initializer='he_normal')(b_c)
    # c_out = classifier_ori(b_c, num_classes=1)
    # c_out = tf.keras.layers.Activation('sigmoid')()
    c_out = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(b_c)
    for _ in range(2):
        c_out = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same")(c_out)

    c_out = tf.keras.layers.Conv2D(1,
                               kernel_size=1,
                               padding='same',
                               activation='sigmoid',
                               kernel_initializer='he_normal')(c_out)

    if train:
        output_b = BilinearInterpolation(input_shape[:2])(b)
        output_c = BilinearInterpolation(input_shape[:2])(c)
        output = [output_b, output_c, c_out]
    else: 
        # output = BilinearInterpolation(input_shape[:2])(c_out)
        output = c_out

    return tf.keras.models.Model(input_, output)
    
