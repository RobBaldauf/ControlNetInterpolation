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
img1 = Image.open('data/genie1.png').resize((768, 640))
img2 = Image.open('data/genie2.jpeg').resize((768, 640))

pose_path = 'data/genie_poses.pk'
if osp.exists(pose_path):
    pose1, pose2 = pickle.load(open(pose_path, 'rb'))
else:
    p1, pose1 = CM.get_pose(img1, return_metadata=True, filter_largest=False)
    Image.fromarray(p1).save('pose1.png')

    canny2 = CM.get_canny(img2)
    Image.fromarray(canny2).save('canny2.png')
    prompt = 'photo of two people, a tall statue on the left looking to the right, a person posing on the right. photorealistic'
    n_prompt = 'blurry, cartoon, painting, 2D'
    out2 = CM.generate(control=canny2, prompt=prompt, n_prompt=n_prompt, mode='canny', guide_scale=20, ctrl_scale=.5)
    Image.fromarray(out2).save('out2.png')
    p2, pose2 = CM.get_pose(out2, return_metadata=True, filter_largest=False)
    Image.fromarray(p2).save('pose2.png')
    
    pickle.dump((pose1, pose2), open(pose_path, 'wb'))

prompt = 'genie Aladdin disney cartoon'
n_prompt = 'text, signature, logo, textured, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality'

# CM.interpolate(img1, img2, cond_path='data/genie500.pt', prompt=prompt, n_prompt=n_prompt, max_steps=.6, optimize_cond=500, ddim_steps=200, num_frames=17, guide_scale=7.5, bias=.1, schedule_type='linear', out_dir='genie')
qc_prompt = 'high quality, male'
qc_neg_prompt = 'text, signature, logo, distorted, ugly, female, blurry'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/genie500.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.25, max_steps=.55, optimize_cond=500, ddim_steps=200, num_frames=17, guide_scale=7.5, schedule_type='linear', out_dir='genie_clip')
