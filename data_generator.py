import numpy as np
import tensorflow as tf
import cv2

from augmentation import PortraitAugment
from preprocess import image_preprocess

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


class DataGeneratorMatting(tf.keras.utils.Sequence):
    'Generate data for Keras'

    def __init__(self, list_IDs, batch_size=32, dim=(512, 512), n_channels=3, shuffle=True, augment=False, train=True, output_div=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = np.array(list_IDs)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        self.train = train
        self.output_div = output_div

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
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

        blank_token = 0

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))

        if "BlankDataset" in img_path:
            blank_token = 1

        if blank_token:
            mask = np.zeros(self.dim, dtype=np.int)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (w, h))

        if self.augment:
            if blank_token:
                img, mask = aug.augment(img, mask, aug_params, mask_transform=False)
            else:
                img, mask = aug.augment(img, mask, aug_params)
            # try :
            #     img, mask = aug.augment(img, mask, aug_params)
            # except : 
            #     print(img_path)
            #     print(mask_path)
        
        if blank_token:
            dil = np.zeros_like(mask)
        else: 

            # for Boundary Attention
            dil = get_edge(mask)
            # try: 
            #     dil = get_edge(mask)
            # except :
            #     print(mask_path)

        # mask thresholding
        # mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)[1]
        mask = mask[:,:,np.newaxis]
        dil = dil[:,:,np.newaxis]

        norm_img = image_preprocess(img)
        norm_mask = mask / 255.
        norm_dil = dil / 255. 
        if self.output_div != 1:
            norm_mask = cv2.resize(norm_mask, (norm_img[1]//self.output_div, norm_img[0]//self.output_div))
            norm_dil = cv2.resize(norm_dil, (norm_img[1]//self.output_div, norm_img[0]//self.output_div))

        return norm_img, norm_mask, norm_dil

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dim[0]//self.output_div, self.dim[1]//self.output_div, 1))
        b = np.empty((self.batch_size, self.dim[0]//self.output_div, self.dim[1]//self.output_div, 1))
        
        # Generate data
        for idx, ID in enumerate(list_IDs_temp):
            # Store sample & uv maskp
            if 'Supervisely' not in ID and "/Custom/img" not in ID:
                mask_ID = ID.split(".p")[0] + "_matte.png"
            else :
                mask_ID = ID.replace("/img/", "/alpha/")

            # if 'Supervisely' in mask_ID:
            #     mask_ID = mask_ID.replace("/img/", "/alpha/")
                # mask_ID = mask_ID.replace(".jpeg", "")

            X[idx], y[idx], b[idx] = self.__get_data(img_path=ID, 
                                               mask_path=mask_ID)             
        return X, [y, b]
