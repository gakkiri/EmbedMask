# EmbedMask
Unofficial implementation for EmbedMask instance segmentation  
office: https://github.com/yinghdb/EmbedMask  
arxiv:  https://arxiv.org/abs/1912.01954  
  
## Log
#### 2020/6/3   
|config|bbox|mask|weight|
|-|:-:|-:|-:|
|MS_R_50_2x.yaml|40.399|34.105|[google drive](https://drive.google.com/file/d/18p5s2NCZwbBNzZnUmfovF9RM1hzxlEX4/view?usp=sharing)|
#### Before 2020/6/3 
The performance is not very stable at present.  
**MS_R_50_2x.yaml**, **box AP 34%** and **seg AP 28%** were reached after a brief training.  
Under the same number of iterations, **MS_X_101_3x.yaml** gets **box AP 37%**, while **seg AP is only 24%**.  
[Here](https://github.com/gakkiri/EmbedMask/blob/master/fcos/modeling/fcos/fcos_outputs.py#L363) I use the CPU implementation, so the FPS is low.  

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


## Results  
### 2020/6/3  
#### MS_R_50_2x.yaml  
![box](https://raw.githubusercontent.com/gakkiri/EmbedMask/master/img/box50.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![seg](https://raw.githubusercontent.com/gakkiri/EmbedMask/master/img/mask50.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)  

## history
I trained about 10 epochs, a V100 takes about 2 days.  
#### MS_X_101_3x.yaml
![box](https://raw.githubusercontent.com/gakkiri/EmbedMask/master/img/bbox_ap.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![seg](https://raw.githubusercontent.com/gakkiri/EmbedMask/master/img/seg_ap.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)

#### MS_R_50_2x.yaml
![box](https://raw.githubusercontent.com/gakkiri/EmbedMask/master/img/box_50.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![seg](https://raw.githubusercontent.com/gakkiri/EmbedMask/master/img/seg_50.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
