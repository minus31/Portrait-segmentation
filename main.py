import os
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf

from data_generator import DataGeneratorMatting
from metrics import *
from network import *
from fast_scnn import fastSCNN

import keras

import warnings
warnings.filterwarnings("ignore")

def get_current_day():
        import datetime
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d')

## Make sure of randomness
np.random.seed(7777)

class SeerSegmentation():

    def __init__(self, config):
    
        self.input_shape = (config.input_shape, int(config.input_shape * 0.75), 3)
        
        if config.train:
            self.batch_size = config.batch_size
            self.nb_epoch = config.nb_epoch
            self.lr = config.lr

        self.finetune = config.finetune

        self.val_ratio = config.val_ratio# default=0.8

        self.checkpoint_path = os.path.join(config.checkpoint_path, get_current_day()) 

        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        
        self.weight_dir = config.weight_dir# default=None

        if config.single_gpu:
            print('\n Specify an GPU \n ')
            gpu_number = input()
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number.strip()

        ##############################################
        # self.img_paths = np.load("./dataset/img_paths_with_supervisely.npy")
        self.img_paths = np.load("./dataset/img_paths_with_supervisely_nosmallobject_k15_withBlankDoubled_and_Custom_doubled.npy")
        
    def build_model(self, train=True):
        return fastSCNN(self.input_shape, train=train)

    def train(self):
        self.model = self.build_model()
        if self.finetune:
            self.model.load_weights(self.weight_dir, by_name=True)
            print('\nload pre-trained model weights\n')
        
        train_params = {
            'dim': self.input_shape[:2],
            'batch_size': self.batch_size,
            'n_channels': self.input_shape[-1],
            'shuffle': True,
            'augment': True,
            'output_div' : 1,
        }

        test_params = {
            'dim': self.input_shape[:2],
            'batch_size': self.batch_size,
            'n_channels': self.input_shape[-1],
            'shuffle': False,
            'augment': False,
            'output_div' : 1,
        }
        
        img_paths = self.img_paths

        self.train_img_paths = np.random.choice(img_paths, int(img_paths.shape[0] * self.val_ratio), replace=False)
        self.test_img_paths = np.setdiff1d(img_paths, self.train_img_paths)

        train_gen = DataGeneratorMatting(self.train_img_paths, **train_params)
        test_gen = DataGeneratorMatting(self.test_img_paths, **test_params)

        """ Training loop """
        STEP_SIZE_TRAIN = len(self.train_img_paths) // train_gen.batch_size
        STEP_SIZE_VAL = len(self.test_img_paths) // test_gen.batch_size

        output_names = []
        for o in self.model.output:
            output_names.append(o.name.split("/")[0])
        print(output_names)

        t0 = time.time()
        for epoch in range(self.nb_epoch):
            opt = tf.keras.optimizers.Adam(lr=self.lr)
            self.model.compile(
                                loss={output_names[0] : ce_dice_focal_combined_loss,
                                    output_names[1] : "binary_crossentropy",},
                                loss_weights=[0.8, 0.2],
                                optimizer=opt,
                                metrics={output_names[0] : [iou_coef, "mse"]})

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
            print(f"\nModel saved with name {model_name}")

            print('\nTraining time for one epoch : %.1f' % ((t2 - t1)))

            if epoch % 100 == 0:
                self.lr = self.lr * 0.5

        print("\nEntire training time has been taken {} ", t2 - t0)

        return None

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--input_shape', type=int, default=256)
    args.add_argument('--nb_epoch', type=int, default=10000)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=0.0045)
    args.add_argument('--val_ratio', type=float, default=0.8)

    args.add_argument('--checkpoint_path', type=str, default="./trained_models")
    args.add_argument('--weight_dir', type=str, default="")
    args.add_argument('--img_path', type=str, default="")

    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--finetune', type=bool, default=False)
    args.add_argument('--single_gpu', type=bool, default=True)

    config = args.parse_args()

    seerSeg = SeerSegmentation(config)

    if config.train:
        seerSeg.train()