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

##### The methods need label transformation #####
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
def rotate_and_scale(img, mask, angle_range, scale_range, mask_transform=True):
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

    if mask_transform:
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
def add_scalar_per_channel(img, mask):
    """
    add scalar to image 

    scalar range : -50 ~ 50
    """
    scalar =  np.random.randint(-50, 50, 3) / 1.
    return np.uint8(np.clip(img + scalar, 0, 255)), mask

@random_activate
def gamma(img, mask, gamma_range):
    if type(gamma_range) == float:
        gamma = gamma_range
    else:
        gamma = np.random.randint(int(gamma_range[0] * 100.0), int(gamma_range[1] * 100.0)) / 100.0
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    img = cv2.LUT(img, table)
    return img, mask
 
@random_activate
def blur(img, mask):
    toss = np.random.choice(np.arange(3))
    # mean blur
    if toss == 0:
        k = np.random.randint(5, 21)
        img = cv2.blur(img, (k, k))
    # median blur
    elif toss == 1:
        k = np.random.choice(np.arange(5, 21, 2))
        img = cv2.medianBlur(img, k)
    # gaussian blur
    else:
        k = np.random.choice(np.arange(5, 21, 2))
        std = np.random.randint(2,5)
        img = cv2.GaussianBlur(img, (k, k), std)
    return img, mask

@random_activate
def occlusion(img, mask):
    # random cut-off
    mod = img.copy()
    num_box = np.random.randint(1, 7)
    k = np.int(img.shape[0] * 0.2)
    for _ in range(num_box):
        w = np.random.randint(0, img.shape[0])
        h = np.random.randint(0, img.shape[0])
        mod[h:h+k, w:w+k, :] = np.ones(mod[h:h+k, w:w+k, :].shape) * 127
    # saliency based occlusion ,,, not implemented
    # grad_kernel = np.array([1, 0, -1])
    # saliency = cv2.filter2D(img, -1, grad_kernel)
    return mod, mask

class PortraitAugment(object):
    """
    Portrait segmentation augmentation.
    """
    def __init__(self):
        pass

    def augment(self, img, mask, param_dict, mask_transform=True):
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
        img, mask = rotate_and_scale(img, mask, param_dict["angle_range"], param_dict["scale_range"], mask_transform=mask_transform)

        # Don't need to transform the labels
        img, mask = add_scalar(img, mask)
        img, mask = add_scalar_per_channel(img, mask)
        img, mask = gamma(img, mask, param_dict["gamma_range"])
        img, mask = blur(img, mask)
        img, mask = occlusion(img, mask)

        return img, mask
    

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def aug_seq():
    sometimes = lambda aug: iaa.Sometimes(0.4, aug)
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5), 
            iaa.Flipud(0.5), 
            sometimes(iaa.CropAndPad(
                percent=(-0.2, 0.2),
                pad_mode="constant",
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-90, 90), 
                shear=(-16, 16), 
                order=[0, 1],
                cval=(0, 255),
                mode="constant" 
            )),
            
            iaa.SomeOf((0, 1),
            [
                iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
                iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
                iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False),
                iaa.Cutout(fill_mode="gaussian", fill_per_channel=True),
                iaa.Cutout(fill_mode="constant", cval=(0, 255), fill_per_channel=0.5),
                iaa.CoarseSaltAndPepper(0.05, size_px=(4, 16))
            ]),

            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), 
                        iaa.AverageBlur(k=(2, 7)), 
                        iaa.MedianBlur(k=(3, 11)), 
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True),
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.AddToHueAndSaturation((-20, 20)),
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.LinearContrast((0.5, 2.0))
                        )
                    ]),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq 

        
