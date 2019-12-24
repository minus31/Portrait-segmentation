from keras import backend as K
import keras
import tensorflow as tf
import numpy as np 

# Metrics for checking portrait segmentation model

# def matting_loss(y_true, y_pred, eps=1e-6):
#     # l_alpha
#     L_alpha = K.mean(K.sqrt(K.pow(y_pred - y_true, 2.) + eps))
    
#     # L_composition
#     fg = K.concatenate((y_true, y_true, y_true), 1)
#     fg_pred = K.concatenate((y_pred, y_pred, y_pred), 1)
#     L_composition = K.mean(K.sqrt(K.pow(fg - fg_pred, 2.) + eps))
    
#     L_p = 0.5 * L_alpha + 0.5 * L_composition
#     return L_p

# def focal_loss(gamma=2.0, alpha=0.25, epsilon=1e-6):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.sum(alpha * K.pow(1. -pt_1, gamma) * K.log(pt_1 + epsilon))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#     return focal_loss_fixed

def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
        return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))

        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss

def ce_dl_combined_loss(y_true, y_pred):
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred))

# works fine --> included in combined
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    return 1 - numerator / denominator

def mse(y_true, y_pred):

    return tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))

def ce_dice_focal_combined_loss(y_true, y_pred):
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))
    
    focal = focal_loss()(y_true, y_pred)
    dl_ce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred))
    mse_ = mse(y_true, y_pred)
    
    return tf.add(tf.add(tf.multiply(focal, 0.1), tf.multiply(dl_ce, 0.1)), tf.multiply(mse_, 0.8))

# the metric
def iou_coef(y_true, y_pred, smooth=1):
    
    threshold = tf.constant(0.5)
    
    y_true = tf.cast(y_true > threshold, dtype=tf.float32)
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

# def mean_iou(y_true, y_pred):

#     prec = []

#     for t in np.arange(0.5, 1.0, 0.05):

#         y_pred_ = tf.to_int32(y_pred > t)
#         score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())

#         with tf.control_dependencies([up_opt]):

#             score = tf.identity(score)

#         prec.append(score)

#     return K.mean(K.stack(prec), axis=0)


# def iou_coef(y_true, y_pred, smooth=1):
#     """
#     IoU = (|X &amp; Y|)/ (|X or Y|)
#     """
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
#     return (intersection + smooth) / (union + smooth)

# def iou_coef_loss(y_true, y_pred):
#     return 1-iou_coef(y_true, y_pred)

# def dice_coef(y_true, y_pred, smooth=1):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
         
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
#     """

#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return 1-dice_coef(y_true, y_pred)