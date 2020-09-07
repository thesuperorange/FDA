import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc

src_img = "demo_images/COCO_val2014_000000069213.jpg"
target_img ="demo_images/Pathway2_3_ch4_00430.jpg" 

im_src = Image.open(src_img).convert('RGB')
im_trg = Image.open(target_img).convert('RGB')

im_src = im_src.resize( (1024,512), Image.BICUBIC )
im_trg = im_trg.resize( (1024,512), Image.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

src_in_trg = src_in_trg.transpose((1,2,0))
scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('demo_images/src_in_tar.png')

