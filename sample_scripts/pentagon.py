import pdb
import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/pentagon1.png').convert('RGB').resize((768, 768))
img2 = Image.open('data/pentagon2.png').convert('RGB').resize((768, 768))

prompt = 'pentagon, building, math, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'lowres, lopsided, disfigured, poorly drawn, low quality'

qc_prompt = 'pentagon, shape, high resolution, ultra HD, simple, elegant, symmetrical'
qc_neg_prompt = 'lines, lowres, distorted, blurry, low quality, messy, fractal'
CM.interpolate_qc(img1, img2, n_choices=5, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/pentagon400.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.35, max_steps=.6, optimize_cond=400, ddim_steps=200, num_frames=17, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='pentagon_clip')
