import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc
import os


img_type =  'val' #'train' #'val'

coco_input_folder = '/home/superorange5/data/coco/'+img_type+'2014'
#MI3_folder = '/home/superorange5/MI3_dataset/MI3_dataset/JPEGImages/'
MI3_folder = '/home/superorange5/MI3_dataset/MI3_dataset_ch4/JPEGImages/'
coco_output_folder = '/home/superorange5/data/coco/FDA0001_ch4__'+img_type+'2014'
BETA = 0.001
if not os.path.isdir(coco_output_folder):
    os.makedirs(coco_output_folder)

MI3_list = os.listdir(MI3_folder)
coco_list = os.listdir(coco_input_folder)

for idx in range(max(len(MI3_list),len(coco_list))):
    print(idx)
    coco_idx = idx % len(coco_list)
    MI3_idx = idx % len(MI3_list)
    
    coco_filename = coco_list[coco_idx]
    MI3_filename = MI3_list[MI3_idx]
    src_img = os.path.join(coco_input_folder,coco_filename)
    target_img = os.path.join(MI3_folder,MI3_filename)


    im_src = Image.open(src_img).convert('RGB')
    im_trg = Image.open(target_img).convert('RGB')


    #im_src = im_src.resize( (640,480))
    im_trg = im_trg.resize( im_src.size)

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=BETA )
    
    #src_in_trg = src_in_trg.transpose((1,2,0))
    scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save(os.path.join(coco_output_folder,coco_filename))
