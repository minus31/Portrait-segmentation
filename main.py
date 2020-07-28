import os
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf

from data_generator import DataGeneratorMatting
from metrics import *
from network import *

import keras

import warnings
warnings.filterwarnings("ignore")

def get_current_day():
        import datetime
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d')

## setting seed
np.random.seed(7777)
# tf.random.set_seed(7777)

class SeerSegmentation():

    def __init__(self, config):
    
        self.input_shape = (config.input_shape, config.input_shape, 3)
        
        if config.train:
            self.batch_size = config.batch_size
            self.nb_epoch = config.nb_epoch
            self.lr = config.lr

        self.finetune = config.finetune
            
        self.val_ratio = config.val_ratio# default=0.8
        self.model_size = config.model_size
        self.checkpoint = config.checkpoint # default=100

        # default="trained_models/{get_current_day()}"
        self.checkpoint_path = os.path.join(config.checkpoint_path, get_current_day()) 
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        
        # if config.finetune or config.infer_single_img:
        self.weight_dir = config.weight_dir# default=None

        ##############################################
        # self.img_paths = np.load("./dataset/img_paths_with_supervisely.npy")
        self.img_paths = np.load("./dataset/img_paths_with_supervisely_nosmallobject_k15_withBlankDoubled_and_Custom_doubled.npy")
        
    def build_model(self, train=True):

        if self.model_size == "big":
            return network_big(input_size=self.input_shape, train=train)

        elif self.model_size == "small":
            return network_small(input_size=self.input_shape, train=train)

    def train(self):

        self.model = self.build_model()

        if self.finetune:
            try : 
                self.model.load_weights(self.weight_dir, by_name=True)
                print('\nload pre-trained model weights\n')

            except Exception as err: 
                print(err)
                print("\n")
                print("^" * 100)
                print("Because of the above error model weights have not loaded")
        output_div = 1.
        train_params = {
            'dim': self.input_shape[:2],
            'batch_size': self.batch_size,
            'n_channels': self.input_shape[-1],
            'shuffle': True,
            'augment': True,
            'train':True,
        }

        test_params = {
            'dim': self.input_shape[:2],
            'batch_size': self.batch_size,
            'n_channels': self.input_shape[-1],
            'shuffle': False,
            'augment': False,
            'train':False,
        }
        
        img_paths = self.img_paths

        self.train_img_paths = np.random.choice(img_paths, int(img_paths.shape[0] * self.val_ratio), replace=False)
        self.test_img_paths = np.setdiff1d(img_paths, self.train_img_paths)

        train_gen = DataGeneratorMatting(self.train_img_paths, **train_params)
        test_gen = DataGeneratorMatting(self.test_img_paths, **test_params)

        opt = tf.keras.optimizers.Adam(lr=self.lr)

        # Freeze parts of network
        # for layer in self.model.layers[:-3]:
        #     layer.trainable = False

        self.model.compile(
                      loss={"output" : ce_dice_focal_combined_loss,
                            "boundary_attention" : "binary_crossentropy",},
                      loss_weights=[0.8, 0.2],
                      optimizer=opt,
                      metrics={"output" : [iou_coef, "mse"]})

        """ Training loop """
        STEP_SIZE_TRAIN = len(self.train_img_paths) // train_gen.batch_size
        STEP_SIZE_VAL = len(self.test_img_paths) // test_gen.batch_size
        t0 = time.time()

        for epoch in range(self.nb_epoch):
            t1 = time.time()
            res = self.model.fit_generator(generator=train_gen,
                                      validation_data=test_gen,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_steps = STEP_SIZE_VAL,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            print(res.history)

            model_name = os.path.join(self.checkpoint_path, str(epoch + 1) + "_" + str(np.round(res.history['val_loss'][0], 2)) + ".h5")
            self.model.save_weights(model_name)
            print(f"Model saved with name {model_name}")

            print('Training time for one epoch : %.1f' % ((t2 - t1)))



            # checkpoint마다 id list를 섞어서 train, Val generator를 새로 생성
            # if (epoch + 1) % self.checkpoint == 0:
                # test_example(self.model, "./result_sample/" + str(epoch) + ".png")
                # print("shuffles the datasets")
                # self.train_img_paths = np.random.choice(img_paths, int(img_paths.shape[0] * self.val_ratio), replace=False)
                # self.test_img_paths = np.setdiff1d(img_paths, self.train_img_paths)

                # train_gen = DataGeneratorMatting(self.train_img_paths, **train_params)
                # test_gen = DataGeneratorMatting(self.test_img_paths, **test_params)

                # self.model.save_weights(os.path.join(self.checkpoint_path, str(epoch + 1) + ".h5"))
                # print("Model saved with name {} ".format(epoch + 1))

        print("Entire training time has been taken {} ", t2 - t0)

        return None

    # def infer_single_img(self, img_path):

    #     self.model = self.build_model(batchnorm=False, train=False)

    #     self.model.load_weights(self.weight_dir, by_name=True)

    #     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #     print(img.shape)
    #     resize_img = cv2.resize(img, self.input_shape[:2][::-1])[np.newaxis,:,:,::-1]
    #     norm_img = resize_img / 255.0
    #     print(norm_img.shape)
    #     pred = self.model.predict(norm_img) * 255.0
    #     print(pred.shape)
    #     pred_modified = pred.squeeze(0).squeeze(-1)
    #     cv2.imwrite("./" + "test"  + img_path.split("/")[-1].split(".")[0] + ".png", pred_modified)

    #     return pred

    # def convert_tflite(self, tflite_name, android=False):
    #     if android : 
    #         self.model = self.build_model_forAndroid(batchnorm=False)
    #     else : 
    #         self.model = self.build_model(batchnorm=False)


    #     # self.model.load_weights(self.weight_dir)

    #     input_names = [node.op.name for node in self.model.inputs]
    #     output_names = [node.op.name for node in self.model.outputs]

    #     print(input_names)
    #     print(output_names)

    #     sess = K.get_session()
    #     converter = tf.lite.TFLiteConverter.from_session(sess, self.model.inputs, self.model.outputs)

    #     tflite_model = converter.convert()
    #     open(tflite_name, "wb").write(tflite_model)
    #     return None



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--input_shape', type=int, default=196)
    args.add_argument('--nb_epoch', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=0.00045)
    args.add_argument('--val_ratio', type=float, default=0.8)

    args.add_argument('--model_size', type=str, default="big")
    args.add_argument('--checkpoint', type=int, default=100)
    args.add_argument('--checkpoint_path', type=str, default="./trained_models")
    args.add_argument('--weight_dir', type=str, default="")
    args.add_argument('--img_path', type=str, default="")
    args.add_argument('--tflite_name', type=str, default="")

    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--finetune', type=bool, default=False)
    args.add_argument('--infer_single_img', type=bool, default=False)
    args.add_argument('--convert', type=bool, default=False)
    args.add_argument('--android', type=bool, default=False)

    config = args.parse_args()

    seerSeg = SeerSegmentation(config)

    if config.train:
        seerSeg.train()

    if config.infer_single_img:
        print("NOT IMPLEMETED")
        # pred = seerSeg.infer_single_img(config.img_path)
        # print(pred)

    if config.convert:
        print("NOT IMPLEMETED")
        # if config.android:
        #     seerSeg.convert_tflite(android=True)
        # else: 
        #     seerSeg.convert_tflite()
