# Portrait-segmentation

#### train command

- Train

`python3 main.py --input_shape=256 --nb_epoch=1000 --batch_size=32 --lr=0.00045 --val_ratio=0.8 --checkpoint=100 --checkpoint_path='./trained_models' --weight_dir="etc.h5" --tflite_name="/Users/hyunkim/Desktop/Segmentation/Portrait-segmentation/tflite/191206_tanh.tflite" --train=True --finetune=False --convert=False --android=False`

* Fine-Tuning

`python3 main.py --input_shape=256 --nb_epoch=1000 --batch_size=32 --lr=0.00045 --val_ratio=0.8 --checkpoint=100 --checkpoint_path='./trained_models' --weight_dir="/home/hyunkim/Portrait-segmentation/trained_models/portrait_seg_matting_256_191017_val_loss_0.0404_val_acc_0.9725_focal_1312.8737.h5" --tflite_name="/Users/hyunkim/Desktop/Segmentation/Portrait-segmentation/tflite/191206_tanh.tflite" --train=True --finetune=True --convert=False --android=False`



##### Todo 

- Capturing the present model output to compare with the better one.

### 0) Problem 

​	1) Edges are inconsistent 

​		- Model Uncertainty - More data, more Training, Photomatric distortion

​	2) Segmenting wrong object

​		- More data

3) 삼성폰에서 Softmax가 잘 작동되지 않는 것

		- 내가 Sigmoid, Softmax, Sigsoftmax  실험 해보기



### 1) Task Specification 

1. Expanding Dataset 

2. Improving pre-processing  (Data Augmentation methods)

3. ? Output post-processing
   - during the process to make the output of the model to the image original resolution



#### Previous model 

`portrait_segmentation_seerslab_256_4channels.tflite` : with softmax

`portrait_segmentation_seerslab_256_without_softmax_4channels_filter_191021_3rd.tflite` : no softmax



## Contribution 

1. 삼성폰에서 Softmax를 썼을 때 이상하게 추론되는 문제 
2. 
3. 

##### Extra1) Concepts

-----

- Trimap : Image with 3 categories, Foreground, Background, Unknown
- Matte : Alpha channel for Object

$$
Image = \alpha Image + (1-\alpha) Image
$$

----

##### Extra2) Log of Editing 

2019.12.03 : Initializing

----

##### Extra3) References

###### • Companies

- Snapchat (USA) https://lensstudio.snapchat.com/api/
- Hyperconnect (KOREA)
- Nalbi (KOREA)

###### • Links

- Annotation tool : https://github.com/abreheret/PixelAnnotationTool
- DeeplabV3 - Keras : https://github.com/bonlime/keras-deeplab-v3-plus
- Image Segmentation Keras : https://github.com/divamgupta/image-segmentation-keras
- Segmentation_models : https://github.com/qubvel/segmentation_models
- Awesome Semantic Segmentation : https://github.com/mrgloom/awesome-semantic-segmentation
- Semantic Segmentation 첫걸음 : [https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb](https://medium.com/hyunjulie/1편-semantic-segmentation-첫걸음-4180367ec9cb)

- Fast Portraits Segmentation : https://github.com/lizhengwei1992/Fast_Portrait_Segmentation
- Portrait Matting : https://github.com/takiyu/portrait_matting/

###### • Tensorflow lite

- https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7

###### • Papers

[1] Ning Xu Brian Price, Scott Cohen, and Thomas Huang. Deep Image Matting. https://arxiv.org/pdf/1703.03872.pdf

[2] Xiaoyong Shen et al. Automatic Portrait Segmentation for Image Stylization. http://xiaoyongshen.me/webpage_portrait/papers/portrait_eg16.pdf

[3] Seokjun Seo et al.(Hyper-Connect) Towards Real-Time Automatic Portrait Matting on Mobile Devices. https://arxiv.org/pdf/1904.03816v1.pdf

[4 *dataset] Farshid Farhat et al. Intelligent Portrait Composition Assistance. http://personal.psu.edu/fuf111/publications/intelligent-portrait-composition-assistance-deep-learning-image-retrieval.pdf



---

