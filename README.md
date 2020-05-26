# EmbedMask
Unofficial implementation for EmbedMask instance segmentation  
office: https://github.com/yinghdb/EmbedMask  
arxiv:  https://arxiv.org/abs/1912.01954  
  
The model with ResNet-101 backbone achieves 31.7 mAP on COCO val2017 set.  
I didn't adjust it carefully. Unfortunately I don't have the computing resources to make it better.

## Install
The code is based on [detectron2](https://github.com/facebookresearch/detectron2). Please check [Install.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for installation instructions.


## Training 
Follows the same way as detectron2.

Single GPU:
```
python train_net.py --config-file configs/EmbedMask/MS_R_101_3x.yaml
```
Multi GPU(for example 8):
```
python train_net.py --num-gpus 8 --config-file configs/EmbedMask/MS_R_101_3x.yaml
```
Please adjust the IMS_PER_BATCH in the config file according to the GPU memory.


## Inference

Single GPU:
```
python train_net.py --config-file configs/EmbedMask/MS_R_101_3x.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
Multi GPU(for example 8):
```
python train_net.py --num-gpus 8 --config-file configs/EmbedMask/MS_R_101_3x.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## Weights
Trained model can be download in https://drive.google.com/file/d/1dQ7ef_ArV1LiUk7LovxSKImYecNu5Jmr/view?usp=sharing

## Results
I trained about 10 epochs using the resnet-101 backbone, a V100 takes about 2 days.  
![box](https://raw.githubusercontent.com/gakkiri/EmbedMask/master/img/bbox_ap.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![box](https://raw.githubusercontent.com/gakkiri/EmbedMask/master/img/seg_ap.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
