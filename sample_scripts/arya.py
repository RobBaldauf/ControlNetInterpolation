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
img1 = Image.open('data/arya1.png').convert('RGB').resize((768, 768))
img2 = Image.open('data/arya2.jpeg').convert('RGB').resize((768, 768))

pose_path = 'data/arya_poses.pk'
if osp.exists(pose_path):
    pose1, pose2 = pickle.load(open(pose_path, 'rb'))
else:
    p1, pose1 = CM.get_pose(img1, return_metadata=True)
    Image.fromarray(p1).save('pose1.png')
    p2, pose2 = CM.get_pose(img2, return_metadata=True)
    Image.fromarray(p2).save('pose2.png')
    
    pickle.dump((pose1, pose2), open(pose_path, 'wb'))
CM.visualize_poses(poses=(pose1, pose2), num_frames=17, out_dir='arya_poses')

prompt = 'arya stark, young fencer, wide face, swordswoman, dramatic portrait'
n_prompt = 'text, signature, logo, watermark, lowres, low quality, blurry'

qc_prompt = 'portrait of arya stark, confident, young, dramatic'
qc_neg_prompt = 'text, signature, logo, watermark, blurry, low quality'
CM.interpolate_qc(img1, img2, controls=(pose1, pose2), qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/arya400.pt', prompt=prompt, n_prompt=n_prompt, optimize_cond=400, min_steps=.3, ddim_steps=200, num_frames=65, guide_scale=7.5, ddim_eta=.1, schedule_type='linear', out_dir='arya_clip')
