import numpy as np 
import cv2

def resize(img, shape):
	"""
	shape must be in order of (w, h) which is opposite of matrix's shape order.
	"""
    h, w = shape
	return cv2.resize(img, (w, h))
# 	return tf.image.resize(img, shape)

def scale(img):
	"""
	scaling 
	"""
	img = (img - np.array([123.68, 116.779, 103.939])) / np.array([58.393, 57.12, 57.375]) # img = (img - tf.constant([123.68, 116.779, 103.939])) / tf.constant([58.393, 57.12, 57.375])
	return img

def image_preprocess(img, input_shape=None):

    if input_shape:
        img = resize(img, input_shape)

	img = scale(img)
    
	return img