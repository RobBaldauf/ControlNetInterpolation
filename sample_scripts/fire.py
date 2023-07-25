import sys
from PIL import Image
import os
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/fire1.png').resize((768, 640))
img2 = Image.open('data/fire2.jpeg').resize((768, 640))

prompt = 'fire in the background, hell, inferno, dog, elmo, cartoon, table, portrait of sds, symmetrical'
n_prompt = 'firefighters, firetruck, people, text, signature, logo, textured, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality'

qc_prompt = 'fire, cartoon, centered, symmetrical'
qc_neg_prompt = 'firefighters, firetruck, text, signature, logo, distorted, ugly, multiple faces, many eyes'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/fire500.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.3, max_steps=.5, optimize_cond=500, ddim_steps=200, num_frames=65, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='fire_clip')
