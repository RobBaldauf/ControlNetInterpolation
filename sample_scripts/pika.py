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
img1 = Image.open('data/pika1.png').convert('RGB').resize((768, 768))
img2 = Image.open('data/pika2.png').convert('RGB').resize((768, 768))

prompt = 'cute cartoon pikachu, high quality, symmetrical face, pokemon, 8k wallpaper, artstation, highly detailed, head facing sds'
n_prompt = 'text, signature, logo, distorted, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality'

# CM.interpolate(img1, img2, cond_path='data/pika300.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=300, ddim_steps=250, num_frames=17, guide_scale=10, bias=.1, retroactive_interp=False, schedule_type='linear', out_dir='pika_300nori')
# CM.interpolate_then_diffuse(img1, img2, cond_path='data/pika300.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=200, ddim_steps=200, num_frames=17, guide_scale=10, schedule_type='linear', out_dir='pika_itd')

qc_prompt = 'portrait of a pikachu, cartoon, centered, symmetrical face'
qc_neg_prompt = 'blurry, motion, distorted, ugly, bad anatomy, weird face, multiple, low quality'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/pika300.pt', optimize_cond=300, min_steps=.3, max_steps=.65, ddim_steps=200, num_frames=65, guide_scale=10, ddim_eta=0.1, schedule_type='linear', out_dir='pika_clip')
