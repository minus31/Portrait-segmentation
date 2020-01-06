import numpy as np
import sklearn.neighbors
import scipy.sparse
import matplotlib.pyplot as plt
import scipy.misc

import cv2
import os

import torch
import torch.nn as nn


import cv2 as cv
from time import time
from PIL import Image

from indexnet.hlmobilenetv2 import hlmobilenetv2

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


IMG_SCALE = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))

STRIDE = 32
RESTORE_FROM = './pretrained/indexnet_matting.pth.tar'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load pretrained model
net = hlmobilenetv2(
        pretrained=False,
        freeze_bn=True, 
        output_stride=STRIDE,
        apply_aspp=True,
        conv_operator='std_conv',
        decoder='indexnet',
        decoder_kernel_size=5,
        indexnet='depthwise',
        index_mode='m2o',
        use_nonlinear=True,
        use_context=True
    )
net = nn.DataParallel(net)
try:
    checkpoint = torch.load(RESTORE_FROM, map_location=torch.device('cpu'))
    pretrained_dict = checkpoint['state_dict']
except:
    raise Exception('Please download the pretrained model!')
net.load_state_dict(pretrained_dict)
net.to(device)

# switch to eval mode
net.eval()

def load_img_mask_pair(img_path):
    
    
    if 'Supervisely' in img_path:
        
        mask_path = img_path.split(".p")[0] + "_matte.png"
        
        mask_path = mask_path.replace("/img/", "/masks_machine/")
        mask_path = mask_path.replace(".jpeg", "")
        
    else:
        mask_path = img_path.replace("/img/", "/mask/")
        
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
    if np.max(mask) == 1:
        mask = mask * 255
    
    if img.shape[0] > 2000:
        img = cv2.resize(img, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024))
    
    return img, mask

def make_trimap(mask, size=(10, 10)):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    mask = mask / 255.

    dilated = cv2.dilate(mask, kernel, iterations=1) * 255
    eroded = cv2.erode(mask, kernel, iterations=1) * 255

    try :
        cnt1 = len(np.where(mask >= 0)[0])
        cnt2 = len(np.where(mask == 0)[0])
        cnt3 = len(np.where(mask == 1)[0])
        
        # print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
        assert(cnt1 == cnt2 + cnt3)
    except :
        _, mask = cv2.threshold(mask, 0.5, 1., cv2.THRESH_BINARY)
        cnt1 = len(np.where(mask >= 0)[0])
        cnt2 = len(np.where(mask == 0)[0])
        cnt3 = len(np.where(mask == 1)[0])
        assert(cnt1 == cnt2 + cnt3)

    try :
        cnt1 = len(np.where(dilated >= 0)[0])
        cnt2 = len(np.where(dilated == 0)[0])
        cnt3 = len(np.where(dilated == 255)[0])
        
        # print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
        assert(cnt1 == cnt2 + cnt3)

    except :
        _, dilated = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
        cnt1 = len(np.where(dilated >= 0)[0])
        cnt2 = len(np.where(dilated == 0)[0])
        cnt3 = len(np.where(dilated == 255)[0])
        assert(cnt1 == cnt2 + cnt3)

    try : 
        cnt1 = len(np.where(eroded >= 0)[0])
        cnt2 = len(np.where(eroded == 0)[0])
        cnt3 = len(np.where(eroded == 255)[0])
        # print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
        assert(cnt1 == cnt2 + cnt3)
    except :
        _, eroded = cv2.threshold(eroded, 127, 255, cv2.THRESH_BINARY)
        cnt1 = len(np.where(eroded >= 0)[0])
        cnt2 = len(np.where(eroded == 0)[0])
        cnt3 = len(np.where(eroded == 255)[0])
        assert(cnt1 == cnt2 + cnt3)

    trimap = dilated.copy()
    
    trimap[((dilated == 255) & (eroded == 0))] = 128

    return trimap

# from INdexNet
def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr

def save_alpha(target_name, alpha):

    cv2.imwrite(target_name, alpha*255)
    
    return 1 

def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv.resize(x1, dsize=(w,h), interpolation=cv.INTER_CUBIC)
    new_x2 = cv.resize(x2, dsize=(w,h), interpolation=cv.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)

    return new_x

def inference(image_path):

    alpha_path = image_path.replace("/img/", "/alpha/")
    image, mask = load_img_mask_pair(img_path)
    trimap = make_trimap(mask, size=(20, 20))
    
    with torch.no_grad():

        trimap = np.expand_dims(trimap, axis=2)
        image = np.concatenate((image, trimap), axis=2)
        
        h, w = image.shape[:2]

        image = image.astype('float32')
        image = (IMG_SCALE * image - IMG_MEAN) / IMG_STD
        image = image.astype('float32')

        image = image_alignment(image, STRIDE)
        inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0))
        #inputs = inputs.cuda()
        
        # inference
        start = time()
        outputs = net(inputs)
        end = time()

        outputs = outputs.squeeze().cpu().numpy()
        alpha = cv.resize(outputs, dsize=(w,h), interpolation=cv.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1) * 255.
        trimap = trimap.squeeze()
        mask = np.equal(trimap, 128).astype(np.float32)
        alpha = (1 - mask) * trimap + mask * alpha

        _, image_name = os.path.split(image_path)
        Image.fromarray(alpha.astype(np.uint8)).save(os.path.join(alpha_path))
        # Image.fromarray(alpha.astype(np.uint8)).show()

        running_frame_rate = 1 * float(1 / (end - start)) # batch_size = 1
        print('framerate: {0:.2f}Hz'.format(running_frame_rate))
        return alpha.astype(np.uint8)

# DATASET_BASE = "./dataset/Superxvisely_person_dataset"
DATASET_BASE = "./dataset/Custom/img"

# dslist = [ds for ds in os.listdir(DATASET_BASE) if "." not in ds]

# dslist[5:]

# # test
# for ds in dslist:
    
#     img_dir = os.path.join(DATASET_BASE, ds, "img")
#     alpha_dir = os.path.join(DATASET_BASE, ds, "alpha")

#     if not os.path.exists(alpha_dir):
#         os.mkdir(alpha_dir)
        
#     img_list = [os.path.join(DATASET_BASE, ds, "img", i) for i in os.listdir(img_dir)]
    
#     for img_path in img_list:
        
        
#         alpha = inference(img_path)
#         print(img_path, "is done")

img_list = [os.path.join(DATASET_BASE, i) for i in os.listdir(DATASET_BASE) if "DS_Store" not in i]

for img_path in img_list:
    alpha_list = os.listdir("./dataset/Custom/alpha")
    if img_path.split("/")[-1] in alpha_list:
        print("Already DONE", img_path)
        continue

    alpha = inference(img_path)
    print(img_path, "is done")