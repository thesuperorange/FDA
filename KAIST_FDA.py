import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc
import os

img_type = 'val'  # 'train' #'val'

mode = 'DAY' #'NIGHT
if mode == 'DAY':
    training_set = ['set00','set01','set02']
else:
    training_set = ['set03', 'set04', 'set05']
KAIST_visible_folder = '/home/superorange5/data/KAIST/KAIST_COMPLETE/visible_images_'+mode
KAIST_thermal_folder = '/home/superorange5/data/KAIST/KAIST_COMPLETE/thermal_images_'+mode

output_folder = '/home/superorange5/data/KAIST/KAIST_FDA0005_' + mode
BETA = 0.005
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

visible_list = os.listdir(KAIST_visible_folder)
thermal_list = os.listdir(KAIST_thermal_folder)

# 1 to 1
for filename in visible_list:
    if any(set_num in filename for set_num in training_set):
        output_img = os.path.join(output_folder, filename)
        src_img = os.path.join(KAIST_visible_folder, filename)
        target_img = os.path.join(KAIST_thermal_folder, filename)

        if not os.path.isfile(output_img):
            im_src = Image.open(src_img).convert('RGB')
            im_trg = Image.open(target_img).convert('RGB')

            # im_src = im_src.resize( (640,480))
            im_trg = im_trg.resize(im_src.size)

            im_src = np.asarray(im_src, np.float32)
            im_trg = np.asarray(im_trg, np.float32)

            im_src = im_src.transpose((2, 0, 1))
            im_trg = im_trg.transpose((2, 0, 1))

            src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=BETA)

            # src_in_trg = src_in_trg.transpose((1,2,0))
            scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save(output_img)
