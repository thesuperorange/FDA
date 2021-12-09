# Fourier domain adaptation for nighttime pedestrian detection using Faster R-CNN
## Method 
* Flowchart
![flowchart](https://user-images.githubusercontent.com/8772677/145319773-3d8a7fa0-4ebe-41a9-a5c2-1cadb0ecbb4f.PNG)

* Step 

Step 1: Apply FFT to source and target images.

Step 2: Replace the low frequency part of the source amplitude with that from the target.

Step 3: Apply inverse FFT to the modified source spectrum.

* Below is the diagram of the proposed 

![Image of FDA](https://github.com/YanchaoYang/FDA/blob/master/demo_images/FDA.png)


## Usage
### FDA
* Convert MI3
```
python ConvertMI3.py
```
### Training
* use fasterRCNN to train (link)

```
python trainval_net.py --dataset coco --net res101 --cuda --mGPUs --bs 16 --nw 8 --lr_decay_step 4 --lr 0.01 --epochs 10
```

## Model
| beta | epoch5 | epoch10 | AP |
| :-----| ----: | :----: | :----: |
| 0.005 | [download](https://superorange.cos.twcc.ai/FDA_model/COCO2MI3_0.005/faster_rcnn_1_5_14657.pth) | [download](https://superorange.cos.twcc.ai/FDA_model/COCO2MI3_0.005/faster_rcnn_1_10_14657.pth) | 75.94|
| 0.01 | [download](https://superorange.cos.twcc.ai/FDA_model/COCO2MI3_0.01/faster_rcnn_1_5_14657.pth) | [download](https://superorange.cos.twcc.ai/FDA_model/COCO2MI3_0.01/faster_rcnn_1_10_14657.pth) | 76.03|
| 0.05 | [download](https://superorange.cos.twcc.ai/FDA_model/COCO2MI3_0.05/faster_rcnn_1_5_14657.pth) | [download](https://superorange.cos.twcc.ai/FDA_model/COCO2MI3_0.05/faster_rcnn_1_10_14657.pth) |72.26|



## Experimental results
* Object detection accuracy v.s β for (a) different channels (illumination
intensities) and (b) various scenes.
![beta_a_1017](https://user-images.githubusercontent.com/8772677/145324337-000b6372-ff42-4e40-9dc3-21b2250a9024.PNG)



## Reference
FDA: Fourier Domain Adaptation for Semantic Segmentation, CVPR 2020. 
[(paper)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf) 
[github](https://github.com/YanchaoYang/FDA)



