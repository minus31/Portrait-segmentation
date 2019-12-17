import numpy as np
import keras
import cv2
from augmentation import PortraitAugment

# Initialize portrait augment
aug = PortraitAugment()
## Define augment params
aug_params = {
    "angle_range": (-45, 45),
    "scale_range": (0.6, 1.5),
    "gamma_range": (0.5, 1.5)
}

class DataGeneratorMatting(keras.utils.Sequence):
    'Generate data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(256, 256), n_channels=3, shuffle=True, augment=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __get_edge(self, mask):

        edge = cv2.Canny(mask, 50, 100)

        k = np.int((mask[mask > 50].shape[0] / (mask.shape[0] * mask.shape[1])) * 50)

        # ksize = (k, k)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        kernel = cv2.getStructuringElement(2, (k, k))

        dil = cv2.dilate(edge, kernel)
        return dil


    def __get_data(self, img_path, mask_path):
        # Load img & mask
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if "Supervisely" in mask_path:
            mask = mask * 255

        # Resize image and mask
        h, w = self.dim
        img = cv2.resize(img, (w, h))
        mask = cv2.resize(mask, (w, h))

        if self.augment:
            try :
                img, mask = aug.augment(img, mask, aug_params)
            except : 
                print(img_path)
                print(mask_path)
        

        # for Boundary Attention
        dil = self.__get_edge(mask)

        # mask thresholding
        # mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)[1]
        mask = mask[:,:,np.newaxis]
        dil = dil[:, :, np.newaxis]
        # Normalize image and mask - normalize 와 Augmentation 순서 다시 고려해보자


        norm_img = img / 255.0
        norm_mask = mask / 255.0
        norm_dil = dil / 255.0

        return norm_img, norm_mask, norm_dil

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], 1))
        b = np.empty((self.batch_size, self.dim[0], self.dim[1], 1))
        # Generate data
        for idx, ID in enumerate(list_IDs_temp):
            # Store sample & uv mask

            mask_ID = ID.split(".p")[0] + "_matte.png"

            if 'Supervisely' in mask_ID:
                mask_ID = mask_ID.replace("/img/", "/masks_machine/")
                mask_ID = mask_ID.replace(".jpeg", "")
            
            X[idx], y[idx], b[idx] = self.__get_data(img_path=ID, 
                                               mask_path=mask_ID)
        return X, [y, b]