import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/rock1.jpeg').resize((768, 768))
img2 = Image.open('data/rock2.jpeg').resize((768, 768))

prompt = 'portrait of Dwayne Johnson, the rock, mountain, highly detailed, symmetrical, photograph, ultra HD'
n_prompt = 'text, signature, logo, textured, lowres, messy, weird face, lopsided, disfigured, low quality'

qc_prompt = 'centered, symmetrical, photorealistic, 3D'
qc_neg_prompt = 'text, signature, logo, distorted, ugly, cartoon, 2D, multiple faces'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/rock500.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=500, ddim_steps=200, num_frames=65, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='rock_clip')
# CM.interpolate_naive(img1, img2, num_frames=17, out_dir='rock_naive')
# CM.interpolate_then_diffuse(img1, img2, cond_path='data/rock500.pt', optimize_cond=500, num_frames=17, ddim_steps=200, min_steps=60, max_steps=110, ddim_eta=.1, out_dir='rock_id')
# CM.denoise_interp_denoise(img1, img2, cond_path='data/rock500.pt', optimize_cond=500, num_frames=17, ddim_steps=200, min_steps=60, max_steps=110, ddim_eta=.1, share_noise=True, out_dir='rock_did')
# CM.denoise_interp_denoise(img1, img2, cond_path='data/rock500.pt', optimize_cond=500, num_frames=17, ddim_steps=200, min_steps=60, max_steps=110, ddim_eta=.1, share_noise=False, out_dir='rock_noshare')
