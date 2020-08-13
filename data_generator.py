import numpy as np
import tensorflow as tf
import cv2
from augmentation import PortraitAugment
from preprocess import *

class DataGeneratorMatting(tf.keras.utils.Sequence):
    'Generate data for Keras'

    def __init__(self, list_IDs, batch_size=32, dim=(224, 168), n_channels=3, shuffle=True, augment=False, output_div=1):
        'Initialization'
        self.list_IDs = np.array(list_IDs)
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.output_div = output_div

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = self.list_IDs[indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __get_data(self, img_path, mask_path):
        # Load img & mask
        h, w = self.dim
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        use_rot_90 = img.shape[0] < img.shape[1]

        if use_rot_90:
            img = rotate90_img(img)
        img = cv2.resize(img, (w, h))
    
        blank_token = 0
        if "BlankDataset" in img_path:
            blank_token = 1

        if blank_token:
            mask = np.zeros(self.dim, dtype=np.int)
        else: 
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if use_rot_90:
                mask = rotate90_img(mask)
            mask = cv2.resize(mask, (w, h))

        if self.augment:
            if blank_token:
                img, mask = aug.augment(img, mask, aug_params, mask_transform=False)
            else:
                img, mask = aug.augment(img, mask, aug_params)
        
        if blank_token:
            dil = np.zeros_like(mask)
        else: 
            dil = get_edge(mask)

        mask = mask[:,:,np.newaxis]
        dil = dil[:,:,np.newaxis]

        norm_img = image_preprocess(img)
        if self.output_div == 1:
            norm_mask = mask / 255.
            norm_dil = dil / 255.
        else: 
            norm_mask = resize(mask, (img.shape[0] // self.output_div, img.shape[1] // self.output_div)) / 255.
            norm_dil = resize(dil, (img.shape[0] // self.output_div, img.shape[1] // self.output_div)) / 255.

        return norm_img, norm_mask, norm_dil

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], 1))
        b = np.empty((self.batch_size, self.dim[0], self.dim[1], 1))
        
        # Generate data
        for idx, ID in enumerate(list_IDs_temp):
            # Store sample & uv maskp
            if 'Supervisely' not in ID and "/Custom/img" not in ID:
                mask_ID = ID.split(".p")[0] + "_matte.png"
            else :
                mask_ID = ID.replace("/img/", "/alpha/")

            X[idx], y[idx], b[idx] = self.__get_data(img_path=ID, 
                                               mask_path=mask_ID)             
        return X, [y, b]


# Initialize portrait augment
aug = PortraitAugment()
## Define augment params
aug_params = {
    "angle_range": (-45, 45),
    "scale_range": (0.6, 1.5),
    "gamma_range": (0.5, 1.5)
}

def get_edge(mask):

    edge = cv2.Canny(mask, 50, 100)
    k = np.int((mask[mask > 50].shape[0] / (mask.shape[0] * mask.shape[1])) * 40)
    if k < 5:
        k = 5
    ksize = (k, k)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)

    dil = cv2.dilate(edge, kernel)

    return dil

def rotate90_img(img):
    # theta = 3 * np.pi / 2
    # R = np.array([
    #                 [np.cos(theta), -np.sin(theta)],
    #                 [np.sin(theta), np.cos(theta)]
    #             ])
    return np.rot90(img)