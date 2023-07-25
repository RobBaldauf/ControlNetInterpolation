import pdb
import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/scar1.png').resize((768, 576))
img2 = Image.open('data/scar2.jpeg').resize((768, 576))

prompt = 'scar from lion king, hamlet, skull, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality'

qc_prompt = 'smiling face, holding a skull, high quality, high resolution, ultra HD'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, low quality'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/scar500.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.35, max_steps=.6, optimize_cond=500, ddim_steps=200, num_frames=17, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='scar_clip')
