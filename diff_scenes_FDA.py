import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc

import os

def FDA(src_img,target_img):
    im_src = Image.open(src_img).convert('RGB')
    im_trg = Image.open(target_img).convert('RGB')

    im_src = im_src.resize( (1024,512), Image.BICUBIC )
    im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.005 )

    src_in_trg = src_in_trg.transpose((1,2,0))
    return src_in_trg
    
    
src_img = "demo_images/COCO_val2014_000000069213.jpg"


foldername = 'MI3_demo_img'
output_folder = 'MI3_FDA_diffscenes'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for file in os.listdir(foldername):
    scene = file.split('_')[0]
    channel = file.split('_')[2]
    output_target = FDA(src_img,os.path.join(foldername,file))
    output_filename = os.path.join(output_folder,scene+"_"+channel+'.jpg')
    
    scipy.misc.toimage(output_target, cmin=0.0, cmax=255.0).save(output_filename)