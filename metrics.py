import tensorflow as tf
import numpy as np 

def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss_ = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss_)

    return loss

def ce_dl_combined_loss(y_true, y_pred):
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred))


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    return 1 - numerator / (denominator + tf.keras.backend.epsilon())

def mse(y_true, y_pred):

    return tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))

def ce_dice_focal_combined_loss(y_true, y_pred):
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - (numerator / (denominator + tf.keras.backend.epsilon())), (-1, 1, 1))
    
    focal = focal_loss()(y_true, y_pred)
    dl_ce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred))
    mse_ = mse(y_true, y_pred)
    return tf.add(tf.add(tf.multiply(focal, 0.7), tf.multiply(dl_ce, 0.2)), tf.multiply(mse_, 0.1))

# the metric
def iou_coef(y_true, y_pred, smooth=1):
    
    threshold = tf.constant(0.5)
    
    y_true = tf.cast(y_true > threshold, dtype=tf.float32)
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1,2,3])
    union = tf.keras.backend.sum(y_true,[1,2,3])+tf.keras.backend.sum(y_pred,[1,2,3])-intersection
    iou = tf.keras.backend.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def identity_mean_loss(y_true, y_pred):
    loss = y_pred[y_pred > 0]
    return tf.keras.backend.mean(loss)

def identity_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_pred - 0 * y_true)

