import pdb
import sys
from PIL import Image
import os, pickle
osp = os.path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
from controlnet import cm

CM = cm.ContextManager()
img1 = Image.open('data/danny1.jpeg').resize((768, 768))
img2 = Image.open('data/danny2.png').resize((768, 768))

pose_path = 'data/danny_poses.pk'
if osp.exists(pose_path):
    pose1, pose2 = pickle.load(open(pose_path, 'rb'))
else:
    p1, pose1 = CM.get_pose(img1, return_metadata=True)
    Image.fromarray(p1).save('pose1.png')

    canny2 = CM.get_canny(img2, lower_bound=180)
    prompt = 'HDR photo of a person standing, photorealistic, ultra HD, detailed'
    n_prompt = 'blurry, cartoon, painting'
    Image.fromarray(canny2).save('canny2.png')
    out2 = CM.img2img(control=canny2, prompt=prompt, n_prompt=n_prompt, init_img=img2, mode='canny', time_frac=0.7)
    Image.fromarray(out2).save('out2.png')
    p2, pose2 = CM.get_pose(out2, return_metadata=True)
    Image.fromarray(p2).save('pose2.png')
    
    pickle.dump((pose1, pose2), open(pose_path, 'wb'))

# os.remove('data/danny400.pt')

prompt = 'portrait of a muscular man, danny devito, high resolution, highly detailed, ultra HD, 4k'
n_prompt = 'text, signature, logo, textured, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality'

qc_prompt = 'high quality, male, high resolution, ultra HD, detailed'
qc_neg_prompt = 'text, signature, logo, pixelated, lowres, distorted, ugly, female, blurry, low quality'
CM.interpolate_qc(img1, img2, controls=(pose1, pose2), qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/danny400.pt', prompt=prompt, n_prompt=n_prompt, min_steps=.25, max_steps=.6, optimize_cond=400, ddim_steps=200, num_frames=65, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='danny_clip')
