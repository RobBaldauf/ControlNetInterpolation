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
img1 = Image.open('data/godzilla1.png').resize((512, 768))
img2 = Image.open('data/godzilla2.png').resize((512, 768))

prompt = 'godzilla, barney, dinosaur, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'lowres, messy, lopsided, disfigured, low quality'

qc_prompt = 'dinosaur, high quality, high resolution, ultra HD'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, low quality'
CM.interpolate_qc(img1, img2, n_choices=3, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/godzilla500.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.35, max_steps=.6, optimize_cond=500, ddim_steps=200, num_frames=17, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='godzilla_clip')
