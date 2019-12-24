# Portrait-segmentation

#### train command

- Train

`python main.py --input_shape=512 --nb_epoch=10000 --batch_size=32 --lr=0.0001 --val_ratio=0.8 --checkpoint=31 --checkpoint_path='./trained_models/lightnet' --weight_dir="./trained_models/initial.h5" --tflite_name="" --train=True --finetune=False --convert=False --android=False`

* Fine-Tuning

`python main.py --input_shape=256 --nb_epoch=10000 --batch_size=32 --lr=0.0001 --val_ratio=0.8 --checkpoint=31 --checkpoint_path='./trained_models/mattingnet/' --weight_dir="./trained_models/mattingnet/supermodel/20191222/580-0.02.h5" --train=True --finetune=True --convert=False --android=False`

- lsSingle image inference 

`python main.py --input_shape=256 --nb_epoch=1 --batch_size=32 --lr=0.0001 --val_ratio=0.8 --checkpoint=10 --checkpoint_path='./trained_models' --weight_dir="./trained_models/20191212/.h5" --img_path="./dataset/selfie/training/00694.png" --infer_single_img=True`

---

### 사용가능 Docker 

- opencv (명령어 적어두기)
- keras
- git 

#### 구축 예정 

```bash
docker run -Pit -u root:root --name dlhk --rm --runtime=nvidia -v /home/hyunkim:/tf/hyunkim -e "0000" -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu-py3
```

> `e` 태그는 비밀번호 설정하는 것

`docker run -Pit --name dlhk --rm --runtime=nvidia -v /home/hyunkim:/tf/hyunkim -e "0000" -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu-py3`

---

scp -r -i ~/.ssh/hyun.pem ./trained_models/20121212/30.h5 hyunkim@35.229.177.132:/home/hyunkim/Portrait-segmentation/trained_models/ ./



### 모델 파일 이름(mattingnet)

portrait_seg_matting_256_191017_val_loss_0.0404_val_acc_0.9725_focal_1312.8737.h5

-> initial.h5

./trained_models/Strongmodels/02-0.09.h5 : k >= 7  인 이미지에 대해서 학습 시키고 Focal loss의 비중을 0.8 두어서 학습하는 val_iou = 0.9289

./trained_models/Strongmodels/25-0.09.h5 : k >= 7인 이미지에 대해 Focal loss의 비중을 0.8 두어 학습 25epoch버전 val_iou = 못 적음...  

./trained_models/Strongmodels/27-0.04.h5 : k >= 7인 이미지에 대해 Focal loss의 비중을 0.8 두고 combined loss 와 Boundary loss를 0.9 : 0.1로 학습 27epoch버전 val_iou = 0.9256

./trained_models/Strongmodels/332-0.03.h5 : PFCN+ dataset 에 대해서만 Focal 0.8 나머지 0.2  boundary 0.1 나머지 0.9의 비율의 Loss로 학습 332 epochs val-iou = 0.9796  

./trained_models/Strongmodels/455-0.03.h5 : PFCN+ dataset 에 대해서만 Focal 0.8 나머지 0.2  boundary 0.1 나머지 0.9의 비율의 Loss로 학습 455 epochs val-iou = 0.9810 

 ./trained_models/Strongmodels/513-0.03.h5 : PFCN+ dataset 에 대해서만 Focal 0.8 나머지 0.2  boundary 0.1 나머지 0.9의 비율의 Loss로 학습 513 epochs val-iou = 0.9820

./trained_models/Strongmodels/580-0.02.h5 : PFCN+ dataset 에 대해서만 Focal 0.8 나머지 0.2  boundary 0.1 나머지 0.9의 비율의 Loss로 학습 580 epochs val-iou = 모르지만 로스가 제일 낮음

----

### 초기 모델 성능 

**`val Matting loss` : 0.0404. `val accuracy` :0.9725. `focal loss`:1312.873**

-----

### Tensorboard 명령

`tensorboard --logdir=./logs --port=8080`



#### PFCN+ DATASET problems

- wrong foreground

![](https://www.dropbox.com/s/6i3txpri9dtjm78/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-12-23%2014.54.53.png?dl=1)

* hand eliminated

![00153](https://www.dropbox.com/s/6uaba9agxzaow8s/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-12-23%2014.52.34.png?dl=1)

#### Applied Augmentation list 

`nlm : non-lable modified`

- [x] arithmetic (nlm)

  - [x] Add
  - [ ] AddElementwise       
  - [ ] AdditiveGaussianNoise
  - [ ] AdditiveLaplaceNoise 
  - [ ] AdditivePoissonNoise 
  - [ ] Multiply              
  - [ ] MultiplyElementwise   
  - [ ] Dropout               
  - [ ] CoarseDropout         
  - [ ] Dropout2d             
  - [ ] TotalDropout          
  - [ ] ReplaceElementwise    
  - [ ] ImpulseNoise          
  - [ ] SaltAndPepper         
  - [ ] CoarseSaltAndPepper   
  - [ ] Salt                  
  - [ ] CoarseSalt            
  - [ ] Pepper                
  - [ ] CoarsePepper          
  - [ ] Invert                
  - [ ] ContrastNormalization 
  - [ ] JpegCompression       




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
- Semantic Human Matting : https://github.com/lizhengwei1992/Semantic_Human_Matting
- Portrait-Segmention : https://github.com/anilsathyan7/Portrait-Segmentation

###### • Tensorflow lite

- https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7

###### • Papers

[1] Ning Xu Brian Price, Scott Cohen, and Thomas Huang. Deep Image Matting. https://arxiv.org/pdf/1703.03872.pdf

[2] Xiaoyong Shen et al. Automatic Portrait Segmentation for Image Stylization. http://xiaoyongshen.me/webpage_portrait/papers/portrait_eg16.pdf

[3] Seokjun Seo et al.(Hyper-Connect) Towards Real-Time Automatic Portrait Matting on Mobile Devices. https://arxiv.org/pdf/1904.03816v1.pdf

[4 *dataset] Farshid Farhat et al. Intelligent Portrait Composition Assistance. http://personal.psu.edu/fuf111/publications/intelligent-portrait-composition-assistance-deep-learning-image-retrieval.pdf

[5] Song-Hai Zhang et al. PortraitNet: Real-time Portrait Segmentation Network for Mobile Device. http://www.yongliangyang.net/docs/mobilePotrait_c&g19.pdf

[6] Xi Chen et al. Boundary-Aware Network for Fast and High-Accuracy Portrait Segmentation. https://arxiv.org/pdf/1901.03814.pdf

<img src="https://www.dropbox.com/s/l5u1lfn5qu6ddto/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-12-17%2016.59.52.png?dl=1">

1. Semantic extraction  : just encoder decoder (replace it with the present model)
2. For Boundary loss  : Binary cross entropy : target - Canny(마스크에서 엣지를 따는 것), multiple tickness of the boundary, 



[7] Qifeng Chen, Dingzeyu Li, Chi-Keung Tang. KNN matting*. http://dingzeyu.li/files/knn-matting-cvpr2012.pdf

----

