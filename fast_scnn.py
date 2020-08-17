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


"""
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

tf.keras.layers.Activation('relu') -> tf.keras.layers.PReLU(shared_axes=[1, 2]) - 느림. -> swish_activation
"""

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

# def bilinear_interpolation(x, size):
#     out = tf.compat.v1.image.resize_bilinear(x, size, align_corners=True)
#     return out

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
    
    x = _DSConv(x, 3, stride)
    x = tf.keras.layers.Concatenate()([x, b])
    x = _DSConv(x, out_ch//4, stride)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(num_classes,
                               kernel_size=1, 
                               kernel_initializer='he_normal')(x)
    if train :
        return x, b
    return x

def fastSCNN(input_shape=(256,192, 3), train=True):
    input_ = tf.keras.layers.Input(shape=input_shape)
    down = learningToDownsample(input_, dw_ch1=32, dw_ch2=48, out_ch=64)
    gf = globalFeatureExtractor(down)
    fus = featureFusionModule(down, gf, out_ch=128)
    cls = classifier(fus, num_classes=1, train=train)
    if train:
        cls, boundary = cls
        output_c = BilinearInterpolation(input_shape[:2])(cls)
        output_b = BilinearInterpolation(input_shape[:2])(boundary)
        output = [output_c, output_b]
    else: 
        output = BilinearInterpolation(input_shape[:2])(cls)
    return tf.keras.models.Model(input_, output)
    
