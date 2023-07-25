import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/gpt1.png').convert('RGB').resize((768, 768))
img2 = Image.open('data/gpt2.jpeg').convert('RGB').resize((768, 768))

prompt = 'simple elegant logo, vector graphics, 2D, symmetrical, centered, high quality'
n_prompt = 'text, textured, lowres, 3D, complicated, lopsided, ugly, low quality'

qc_prompt = 'simple elegant logo, vector graphics, 2D, symmetrical'
qc_neg_prompt = 'text, textured, overlay, blurry, 3D, complicated, lopsided, ugly, low quality'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/gpt200.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.3, max_steps=.8, optimize_cond=200, ddim_steps=200, num_frames=65, guide_scale=7.5, ddim_eta=.3, schedule_type='linear', out_dir='gpt_clip')
