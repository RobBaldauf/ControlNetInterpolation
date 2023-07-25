import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/thanos1.jpeg').resize((832, 512))
img2 = Image.open('data/thanos2.png').resize((832, 512))

prompt = 'superhero, supervillain, cartoon, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'lowres, messy, weird face, lopsided, disfigured, low quality, photo'

qc_prompt = 'portrait, cartoon, superhero, detailed, high quality'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, photo, low quality, multiple faces'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/thanos500.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.3, max_steps=.5, optimize_cond=500, ddim_steps=200, num_frames=17, guide_scale=7.5, ddim_eta=0, schedule_type='linear', out_dir='thanos_clip')
