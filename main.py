
import numpy as np
import keras
from keras.layers import *
from keras.models import *
from keras.layers import Conv2D, SeparableConv2D, Conv2DTranspose, add, ReLU, Dropout, Reshape, Permute
from keras.activations import sigmoid
from keras.utils.generic_utils import CustomObjectScope
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau

from data_generator import DataGeneratorMatting
from metrics import *
from models import *
import os
import time
import argparse

import cv2

def get_current_day():
        import datetime
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d')


class SeerSegmentation():

    def __init__(self, config):

        self.input_shape = (config.input_shape, config.input_shape, 3)
        
        if config.train:
            self.batch_size = config.batch_size
            self.nb_epoch = config.nb_epoch
            self.lr = config.lr
            
        self.val_ratio = config.val_ratio# default=0.8
        self.checkpoint = config.checkpoint # default=100

        self.checkpoint_path = os.path.join(config.checkpoint_path, get_current_day()) # default="trained_models/{get_current_day()}"
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        
        ##############################################  < 고쳐야함
        if config.finetune or config.infer_single_img:
            self.weight_dir = config.weight_dir# default=None

        if config.convert:
            self.weight_dir = config.weight_dir# default=None
        ##############################################
        self.img_paths = np.load("./dataset/selfie/img_paths.npy")
        
    def build_model(self, batchnorm=False):

        return matting_net(input_size=self.input_shape, batchnorm=batchnorm, android=False)

    def build_model_forAndroid(self, batchnorm=False):

        return matting_net(input_size=(self.input_shape[0], self.input_shape[1], 4), batchnorm=batchnorm, android=True)

    def train(self, finetune=False):

        self.model = self.build_model(batchnorm=True)
        # if finetune:
        #     self.model.load_weights(self.weight_dir)

        train_params = {
            'dim': self.input_shape[:2],
            'batch_size': self.batch_size,
            'n_channels': self.input_shape[-1],
            'shuffle': True,
            'augment': True,
        }

        test_params = {
            'dim': self.input_shape[:2],
            'batch_size': self.batch_size,
            'n_channels': self.input_shape[-1],
            'shuffle': True,
            'augment': False,
        }
        
        img_paths = self.img_paths
        self.train_img_paths = np.random.choice(img_paths, int(img_paths.shape[0] * self.val_ratio), replace=False)
        self.test_img_paths = np.setdiff1d(img_paths, self.train_img_paths)

        train_gen = DataGeneratorMatting(self.train_img_paths, **train_params)
        test_gen = DataGeneratorMatting(self.test_img_paths, **test_params)


        opt = keras.optimizers.adam(lr=self.lr, amsgrad=True)

        self.model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=[iou_coef, focal_loss()])

        """ Callback """
        monitor = 'loss'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
        """Callback for Tensorboard"""
        tb = tf.keras.callbacks.TensorBoard(log_dir="./logs/", update_freq='batch')

        """ Training loop """
        STEP_SIZE_TRAIN = len(self.train_img_paths) // train_gen.batch_size
        STEP_SIZE_VAL = len(self.test_img_paths) // test_gen.batch_size
        t0 = time.time()

        for epoch in range(self.nb_epoch):
            t1 = time.time()
            res = self.model.fit_generator(generator=train_gen,
                                      validation_data=test_gen,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr, tb],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            
            print(res.history)
            
            print('Training time for one epoch : %.1f' % ((t2 - t1)))

            # step 마다 id list를 섞어서 train, Val generator를 새로 생성
            self.train_img_paths = np.random.choice(img_paths, int(img_paths.shape[0] * self.val_ratio), replace=False)
            self.test_img_paths = np.setdiff1d(img_paths, self.train_img_paths)

            train_gen = DataGeneratorMatting(self.train_img_paths, **train_params)
            test_gen = DataGeneratorMatting(self.test_img_paths, **test_params)

            if epoch % self.checkpoint == 0:
                self.model.save_weights(os.path.join(self.checkpoint_path, str(epoch)))

        print("Entire training time has been taken {} ", t2 - t0)

        return 1

    def infer_single_img(self, img_path):
        self.model = self.build_model(batchnorm=False)
        # self.model.load_weights(self.weight_dir)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[np.newaxis,:,:,::-1]
        resize_img = cv2.resize(img, self.input_shape[:2][::-1]) 
        norm_img = resize_img / 255.0
        pred = self.model.predict(norm_img) * 255.0

        cv2.imwrite("./" + "test"  + img_path.split("/")[-1].split(".")[0] + "jpg", pred)

        return pred

    def convert_tflite(tflite_name, android=False):
        if android : 
            self.model = self.build_model_forAndroid(batchnorm=False)
        else : 
            self.model = self.build_model(batchnorm=False)


        # self.model.load_weights(self.weight_dir)

        input_names = [node.op.name for node in self.model.inputs]
        output_names = [node.op.name for node in self.model.outputs]

        print(input_names)
        print(output_names)

        sess = K.get_session()
        converter = tf.lite.TFLiteConverter.from_session(sess, self.model.inputs, self.model.outputs)

        tflite_model = converter.convert()
        open(tflite_name, "wb").write(tflite_model)
        return None



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--input_shape', type=int, default=256)
    args.add_argument('--nb_epoch', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=0.00045)
    args.add_argument('--val_ratio', type=float, default=0.8)

    args.add_argument('--checkpoint', type=int, default=100)
    args.add_argument('--checkpoint_path', type=str, default="./trained_models")
    args.add_argument('--weight_dir', type=str, default="")
    args.add_argument('--img_path', type=str, default="")
    args.add_argument('--tflite_name', type=str, default="")

    args.add_argument('--train', type=bool, default=False)
    args.add_argument('--finetune', type=bool, default=False)
    args.add_argument('--infer_single_img', type=bool, default=False)
    args.add_argument('--convert', type=bool, default=False)
    args.add_argument('--android', type=bool, default=False)

    config = args.parse_args()


    
    seerSeg = SeerSegmentation(config)

    if config.train:
        seerSeg.train(config.finetune)

    if config.infer_single_img:
        seerSeg.infer_single_img(config.img_path)

    if config.convert:
        if config.android:
            seerSeg.convert_tflite(android=True)
        else: 
            seerSeg.convert_tflite()
