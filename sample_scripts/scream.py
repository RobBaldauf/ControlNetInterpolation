import pdb
import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/scream1.jpg').convert('RGB').resize((640, 768))
img2 = Image.open('data/scream2.png').convert('RGB').resize((640, 768))

# pose_path = 'data/scream_poses.pk'
# if osp.exists(pose_path):
#     os.remove(pose_path)
# if osp.exists(pose_path):
#     pose1, pose2 = pickle.load(open(pose_path, 'rb'))
# else:
#     p1, pose1 = CM.get_pose(img1, return_metadata=True)
#     Image.fromarray(p1).save('pose1.png')

#     canny2 = CM.get_canny(img2, lower_bound=350)
#     prompt = 'photo of a person standing, expressive face, the scream, photorealistic'
#     n_prompt = 'blurry, cartoon, painting'
#     Image.fromarray(canny2).save('canny2.png')
#     out2 = CM.generate(control=canny2, prompt=prompt, n_prompt=n_prompt, mode='canny', guide_scale=20, ctrl_scale=.5, ddim_steps=100)
#     Image.fromarray(out2).save('out2.png')
#     p2, pose2 = CM.get_pose(out2, return_metadata=True)
#     Image.fromarray(p2).save('pose2.png')

#     pickle.dump((pose1, pose2), open(pose_path, 'wb'))


prompt = 'the scream, movie poster, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality'

qc_prompt = 'high quality, high resolution, ultra HD, detailed'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, low quality'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/scream500.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.4, max_steps=.6, optimize_cond=500, ddim_steps=200, num_frames=17, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='scream_clip')
