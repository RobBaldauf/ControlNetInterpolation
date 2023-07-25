import pdb
import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/gandalf1.png').convert('RGB').resize((768, 448))
img2 = Image.open('data/gandalf2.jpeg').convert('RGB').resize((768, 448))

prompt = 'portrait of gandalf, portrait of santa claus, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'lowres, lopsided, disfigured, low quality'

qc_prompt = 'gandalf, santa, shape, high resolution, ultra HD, photo, portrait'
qc_neg_prompt = 'lowres, distorted, blurry, low quality, drawing'
CM.interpolate_qc(img1, img2, n_choices=5, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/gandalf300.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.3, max_steps=.5, optimize_cond=300, ddim_steps=200, num_frames=17, guide_scale=7.5, ddim_eta=0, schedule_type='linear', out_dir='gandalf_clip')
