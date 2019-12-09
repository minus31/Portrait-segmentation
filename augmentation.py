import cv2
import numpy as np

# Scale / Rotation / Color / Occlusion
# * Scale (0.6, 0.8, 1.2, 1.5)
# * Rotation (-45, -22, 22, 45)
# * Color gamma Variation (0.5, 0.8, 1.2, 1.5)

class PortraitAugment(object):
    """
    Portrait segmentation augmentation.
    """
    def __init__(self):
        pass

    def augment(self, img, mask, angle_range, scale_range, gamma_range):
        """
        Do image and mask augment.
        Args:
            img: a numpy type
            mask: a numpy type
            angle_range: range for randomly rotation.  eg: 45 or (-45, 45)
            scale_range: scale img, mask.  eg: 0.5 or (0.5, 1.5)
            gamma_range: range for color gamma correction.  eg: 0.6 or (0.6, 1.5)
        Return:
            an image and mask with target size will be return
        Raises:
            No
        """
        img, mask = self.__hflip(img, mask, run_prob=0.5)
        img, mask = self.__rotate_and_scale(img, mask, angle_range, scale_range)
        img = self.__gamma(img, gamma_range)

        return img, mask
    
    def __hflip(self, img, mask, run_prob=0.5):
        if np.random.rand() < run_prob:
            return img, mask
        img = np.fliplr(img)
        mask = np.fliplr(np.expand_dims(mask, axis=-1))
        return img, mask.squeeze()

    def __vflip(self, img, mask, run_prob=0.5):
        if np.random.rand() < run_prob:
            return img, mask
        img = np.flipud(img)
        mask = np.flipud(np.expand_dims(mask, axis=-1))
        return img, mask.squeeze()

    def __rotate_and_scale(self, img, mask, angle_range, scale_range):
        if type(angle_range) == int:
            angle = angle_range
        else:
            angle = np.random.randint(int(angle_range[0]), int(angle_range[1]))

        if type(scale_range) == float:
            scale = scale_range
        else:
            scale = np.random.randint(int(scale_range[0] * 100.0), int(scale_range[1] * 100.0)) / 100.0
        # For image
        h, w = img.shape[:2]
        c_x = w / 2
        c_y = h / 2
        M = cv2.getRotationMatrix2D((c_x, c_y), angle, scale)
        img = cv2.warpAffine(img, M, (w, h))

        # For mask
        h, w = mask.shape[:2]
        c_x = w / 2
        c_y = h / 2
        M = cv2.getRotationMatrix2D((c_x, c_y), angle, scale)
        mask = cv2.warpAffine(mask, M, (w, h))

        return img, mask

    def __gamma(self, img, gamma_range):
        if type(gamma_range) == float:
            gamma = gamma_range
        else:
            gamma = np.random.randint(int(gamma_range[0] * 100.0), int(gamma_range[1] * 100.0)) / 100.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        img = cv2.LUT(img, table)
        return img
