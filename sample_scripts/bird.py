import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/bird1.png').convert('RGB').resize((768, 768))
img2 = Image.open('data/bird2.png').convert('RGB').resize((768, 768))

prompt = 'cartoon bird, ultra HD, logo, simple, 2D, bright'
n_prompt = 'text, signature, textured, lowres, 3D, messy, complicated, lopsided, disfigured, low quality'

qc_prompt = 'cartoon bird, centered, simple, 2D'
qc_neg_prompt = 'text, signature, complicated, heavily textured, distorted, ugly, 3D'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/bird500.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.35, max_steps=.65, optimize_cond=500, ddim_steps=200, num_frames=65, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='bird_clip')
