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
img1 = Image.open('data/king1.jpg').resize((768, 768))
img2 = Image.open('data/king2.jpeg').resize((768, 768))

prompt = 'king, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'lowres, lopsided, disfigured, low quality'

qc_prompt = 'face, king, high quality, high resolution, ultra HD, simple, elegant'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, low quality, messy'
CM.interpolate_qc(img1, img2, n_choices=7, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/king400.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.45, max_steps=.6, optimize_cond=400, ddim_steps=200, num_frames=17, guide_scale=7.5, ddim_eta=.05, schedule_type='linear', zoom_per_frame=1.04, dy=1.5, out_dir='king_clip')
