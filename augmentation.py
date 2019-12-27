import cv2
import numpy as np

# Scale / Rotation / Color / Occlusion
# * Scale (0.6, 0.8, 1.2, 1.5)
# * Rotation (-45, -22, 22, 45)
# * Color gamma Variation (0.5, 0.8, 1.2, 1.5)


def random_activate(aug_func):

    def wrapper(*args, **kwargs):
        if np.random.choice([0, 1]):
            # print("transformed")
            return aug_func(*args, **kwargs)
        else:
            # print("NOT transformed")
            return identity_img(*args, **kwargs)

    return wrapper

def identity_img(img, mask, *args, **kwargs) :
        return img, mask

##### The methods needabel transformation #####
@random_activate
def hflip(img, mask):

    img = np.fliplr(img)
    mask = np.fliplr(np.expand_dims(mask, axis=-1))
    return img, mask.squeeze()

@random_activate
def vflip(img, mask):
    
    img = np.flipud(img)
    mask = np.flipud(np.expand_dims(mask, axis=-1))
    return img, mask.squeeze()

@random_activate
def rotate_and_scale(img, mask, angle_range, scale_range):
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
    # h, w = mask.shape[:2]
    # c_x = w / 2
    # c_y = h / 2
    # M = cv2.getRotationMatrix2D((c_x, c_y), angle, scale)
    mask = cv2.warpAffine(mask, M, (w, h))

    return img, mask


##### The methods don't need label transformation #####
@random_activate
def add_scalar(img, mask):
    """
    add scalar to image 

    scalar range : -50 ~ 50
    """
    scalar =  np.float(np.random.randint(-50, 50))

    return np.uint8(np.clip(img + scalar, 0, 255)), mask

@random_activate
def gamma(img, mask, gamma_range):
    if type(gamma_range) == float:
        gamma = gamma_range
    else:
        gamma = np.random.randint(int(gamma_range[0] * 100.0), int(gamma_range[1] * 100.0)) / 100.0
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    img = cv2.LUT(img, table)ㅌ
    return img, mask
 


class PortraitAugment(object):
    """
    Portrait segmentation augmentation.
    """
    def __init__(self):
        pass

    def augment(self, img, mask, param_dict):
        """
        Do image and mask augment.
        Args:
            img: a numpy type
            mask: a numpy type
            param_dict : 
                - angle_range: range for randomly rotation.  eg: 45 or (-45, 45),
                - scale_range: scale img, mask.  eg: 0.5 or (0.5, 1.5),
                - gamma_range: range for color gamma correction.  eg: 0.6 or (0.6, 1.5),
        Return:
            an image and mask with target size will be return
        Raises:
            No
        """

        # Need to transform the labels
        img, mask = hflip(img, mask)
        img, mask = vflip(img, mask)
        img, mask = rotate_and_scale(img, mask, param_dict["angle_range"], param_dict["scale_range"])

        # Don't need to transform the labels
        img, mask = add_scalar(img, mask)
        img, mask = gamma(img, mask, param_dict["gamma_range"])

        return img, mask
    
    