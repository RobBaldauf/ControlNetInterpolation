import sys
from PIL import Image
import os
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/chess1.png').convert('RGB').resize((768, 512))
img2 = Image.open('data/chess2.jpeg').convert('RGB').resize((768, 512))

prompt = 'chess pieces, coronation, highly detailed, perfect composition, 2D, painting, flat'
n_prompt = 'text, signature, logo, distorted, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, 3D'

qc_prompt = 'highly detailed, perfect composition, 2D'
qc_neg_prompt = 'text, signature, logo, distorted, ugly, bad, weird, 3D'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/chess400.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=400, min_steps=.35, max_steps=.6, ddim_steps=200, num_frames=17, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='chess_clip')
