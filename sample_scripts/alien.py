import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/alien1.png').convert('RGB').resize((768, 640))
img2 = Image.open('data/alien2.png').convert('RGB').resize((768, 640))

prompt = 'space needle, nighttime, UFO, photograph, ultra HD'
n_prompt = 'text, signature, logo, watermark, lowres, complicated, low quality'

qc_prompt = 'nighttime, photograph'
qc_neg_prompt = 'text, signature, logo, watermark, complicated, low quality'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/alien500.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=500, ddim_steps=200, num_frames=17, guide_scale=7.5, schedule_type='linear', out_dir='alien_clip')
