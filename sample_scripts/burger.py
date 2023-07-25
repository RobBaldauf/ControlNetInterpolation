import pdb
import sys
from PIL import Image
import os, pickle
osp = os.path
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/burger1.jpeg').resize((768, 576))
img2 = Image.open('data/burger2.jpeg').resize((768, 576))

prompt = 'advertisement, burger, medicine, photorealistic, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'lowres, messy, lopsided, disfigured, poorly drawn, low quality'

qc_prompt = 'TV advertisement, high quality, high resolution, ultra HD, clean'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, low quality, messy'
CM.interpolate_qc(img1, img2, n_choices=4, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/burger400.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.35, max_steps=.6, optimize_cond=400, ddim_steps=200, num_frames=65, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='burger_clip')
