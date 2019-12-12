from keras import backend as K
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

def mean_iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):
        
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score)

        prec.append(score)

    return K.mean(K.stack(prec), axis=0)


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